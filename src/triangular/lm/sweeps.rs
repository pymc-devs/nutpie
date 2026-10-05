//! The sweeps over one tile: the local kernels for every variable, coupled
//! by the triangular solves. For the names, see the notation in [`super`].

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

use super::conditioner::{Cotangent, Scratch, UnitState};
use super::simd::store;
use super::tiles::{Data, LinearTile, LinearTileMut, Linearization};
use super::FisherResiduals;

/// One tile's draws and linearization, as vectors.
pub(super) struct Tile<S: Simd> {
    /// The map's inputs and the log density's gradients, per variable.
    y: Vec<S::f64s>,
    g: Vec<S::f64s>,
    lin: LinearTile<S>,
}

/// The primal's intermediates.
struct PrimalWork<S: Simd> {
    /// `L`, per edge.
    edge_l: Vec<S::f64s>,
    /// `grad_y log_det`, per variable.
    grad_log_det: Vec<S::f64s>,
}

/// The pushforward's tangents.
struct TangentWork<S: Simd> {
    /// Per variable.
    x_dot: Vec<S::f64s>,
    /// The right-hand side of `J^T w_dot = ...`, then `w_dot`.
    w_dot: Vec<S::f64s>,
    /// `A_dot` and `L_dot`, per edge.
    a_dot: Vec<S::f64s>,
    l_dot: Vec<S::f64s>,
}

/// The pullback's cotangents.
struct CotangentWork<S: Simd> {
    /// `J^{-1}` of the residual cotangents.
    z: Vec<S::f64s>,
    /// `A_bar` and `L_bar`, per edge.
    cot_a: Vec<S::f64s>,
    cot_l: Vec<S::f64s>,
}

/// Per-thread scratch for the sweeps over one tile.
pub(super) struct TileScratch<S: Simd> {
    tile: Tile<S>,
    units: UnitState<S>,
    local: Scratch<S>,
    primal: PrimalWork<S>,
    tangent: TangentWork<S>,
    cotangent: CotangentWork<S>,
    /// The tile's residuals, or their cotangents.
    pub(super) residual: Vec<S::f64s>,
}

impl<S: Simd> TileScratch<S> {
    pub(super) fn new(simd: S, problem: &FisherResiduals) -> Self {
        let zero = S::f64s::splat(simd, 0.0);
        let (n_var, n_edge) = (problem.n_var(), problem.n_edge());
        let by_var = || vec![zero; n_var];
        let by_edge = || vec![zero; n_edge];
        Self {
            tile: Tile {
                y: by_var(),
                g: by_var(),
                lin: LinearTile::new(simd, n_var, n_edge),
            },
            units: problem.unit_state(simd),
            local: problem.conditioner.scratch(simd),
            primal: PrimalWork {
                edge_l: by_edge(),
                grad_log_det: by_var(),
            },
            tangent: TangentWork {
                x_dot: by_var(),
                w_dot: by_var(),
                a_dot: by_edge(),
                l_dot: by_edge(),
            },
            cotangent: CotangentWork {
                z: by_var(),
                cot_a: by_edge(),
                cot_l: by_edge(),
            },
            residual: vec![zero; problem.n_residuals()],
        }
    }
}

impl FisherResiduals {
    /// Loads tile `tile`'s draws and linearization into `s`.
    #[simd]
    pub(super) fn load_tile<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        tile: usize,
        s: &mut TileScratch<S>,
    ) {
        lin.data.y.load(simd, tile, &mut s.tile.y);
        s.tile.lin.load(simd, lin, tile);
    }

    /// Solves `J^T w = b` in place (back substitution, children first).
    #[simd]
    fn solve_transpose<S: Simd>(
        &self,
        _: S,
        edge_a: &[S::f64s],
        delta: &[S::f64s],
        b: &mut [S::f64s],
    ) {
        for i in (0..self.n_var()).rev() {
            let w_i = b[i] / delta[i];
            b[i] = w_i;
            for (e, p) in self.graph.entries(i) {
                b[p] -= edge_a[e] * w_i;
            }
        }
    }

    /// Solves `J z = b` in place (forward substitution, parents first).
    #[simd]
    fn solve<S: Simd>(&self, _: S, edge_a: &[S::f64s], delta: &[S::f64s], b: &mut [S::f64s]) {
        for i in 0..self.n_var() {
            let mut acc = b[i];
            for (e, p) in self.graph.entries(i) {
                acc -= edge_a[e] * b[p];
            }
            b[i] = acc / delta[i];
        }
    }

    /// The primal of one tile: its residuals into `out` and its
    /// linearization into `out_lin`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn primal_tile<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        data: &Data,
        tile: usize,
        out_lin: LinearTileMut<'_>,
        out: &mut [f64],
        s: &mut TileScratch<S>,
    ) {
        let n_var = self.n_var();
        data.y.load(simd, tile, &mut s.tile.y);
        data.g.load(simd, tile, &mut s.tile.g);
        let TileScratch {
            tile:
                Tile {
                    y,
                    g,
                    lin:
                        LinearTile {
                            edge_a,
                            delta,
                            x,
                            w,
                        },
                },
            units,
            local,
            primal: PrimalWork {
                edge_l,
                grad_log_det,
            },
            residual,
            ..
        } = s;
        grad_log_det.fill(S::f64s::splat(simd, 0.0));

        for i in 0..n_var {
            self.gather_parents(i, y, &mut units.y_parents);
            let theta_i = self.params_of(theta, i);
            self.conditioner.evaluate(simd, theta_i, units);
            let edges = self.graph.edges(i);
            let (x_i, delta_i, mu_i) = self.conditioner.local_primal(
                simd,
                theta_i,
                units,
                y[i],
                &mut edge_a[edges.clone()],
                &mut edge_l[edges],
                local,
            );
            x[i] = x_i;
            delta[i] = delta_i;
            grad_log_det[i] += mu_i;
            for (e, p) in self.graph.entries(i) {
                grad_log_det[p] += edge_l[e];
            }
        }

        for i in 0..n_var {
            w[i] = g[i] - grad_log_det[i];
        }
        self.solve_transpose(simd, edge_a, delta, w);

        let scale = data.scale();
        for i in 0..n_var {
            residual[i] = (x[i] + w[i]) * scale;
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.graph.edges(i) {
                    residual[n_var + e] = (edge_l[e] - x[i] * edge_a[e]) * (scale * reg);
                }
            }
        }

        store(simd, residual, out);
        s.tile.lin.store(simd, out_lin);
    }

    /// `J v` for one tile, already loaded into `s`, into `s.residual`.
    #[simd]
    pub(super) fn pushforward_tile<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        v: &[f64],
        s: &mut TileScratch<S>,
    ) {
        let n_var = self.n_var();
        let TileScratch {
            tile:
                Tile {
                    y,
                    lin:
                        LinearTile {
                            edge_a,
                            delta,
                            x,
                            w,
                        },
                    ..
                },
            units,
            local,
            tangent:
                TangentWork {
                    x_dot,
                    w_dot,
                    a_dot,
                    l_dot,
                },
            residual,
            ..
        } = s;
        w_dot.fill(S::f64s::splat(simd, 0.0));

        for i in 0..n_var {
            self.gather_parents(i, y, &mut units.y_parents);
            let theta_i = self.params_of(&lin.theta, i);
            self.conditioner.evaluate(simd, theta_i, units);
            let edges = self.graph.edges(i);
            let (xd, delta_dot, mu_dot) = self.conditioner.local_pushforward(
                simd,
                theta_i,
                self.params_of(v, i),
                units,
                y[i],
                &mut a_dot[edges.clone()],
                &mut l_dot[edges],
                local,
            );
            x_dot[i] = xd;
            // J^T w_dot = -(grad_y log_det)_dot - J_dot^T w
            w_dot[i] -= mu_dot + delta_dot * w[i];
            for (e, p) in self.graph.entries(i) {
                w_dot[p] -= l_dot[e] + a_dot[e] * w[i];
            }
        }
        self.solve_transpose(simd, edge_a, delta, w_dot);

        let scale = lin.data.scale();
        for i in 0..n_var {
            residual[i] = (x_dot[i] + w_dot[i]) * scale;
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.graph.edges(i) {
                    residual[n_var + e] =
                        (l_dot[e] - x_dot[i] * edge_a[e] - x[i] * a_dot[e]) * (scale * reg);
                }
            }
        }
    }

    /// Accumulates `J^T r_bar` for one tile, already loaded into `s` with
    /// `r_bar` in `s.residual`, into `grad`.
    #[simd]
    pub(super) fn pullback_tile<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        s: &mut TileScratch<S>,
        grad: &mut [S::f64s],
    ) {
        let n_var = self.n_var();
        let scale = lin.data.scale();
        let TileScratch {
            tile:
                Tile {
                    y,
                    lin:
                        LinearTile {
                            edge_a,
                            delta,
                            x,
                            w,
                        },
                    ..
                },
            units,
            local,
            cotangent: CotangentWork { z, cot_a, cot_l },
            residual: r_bar,
            ..
        } = s;

        for i in 0..n_var {
            z[i] = r_bar[i] * scale;
        }
        self.solve(simd, edge_a, delta, z);

        for i in 0..n_var {
            self.gather_parents(i, y, &mut units.y_parents);
            let edges = self.graph.edges(i);
            let mut x_bar = r_bar[i] * scale;
            for (e, p) in self.graph.entries(i) {
                let z_p = z[p];
                cot_a[e] = -(w[i] * z_p);
                cot_l[e] = -z_p;
                if let Some(reg) = self.regularization {
                    let o = r_bar[n_var + e] * (scale * reg);
                    x_bar -= o * edge_a[e];
                    cot_a[e] -= x[i] * o;
                    cot_l[e] += o;
                }
            }
            let cot = Cotangent {
                x: x_bar,
                delta: -(w[i] * z[i]),
                mu: -z[i],
                edge_a: &cot_a[edges.clone()],
                edge_l: &cot_l[edges],
            };
            let theta_i = self.params_of(&lin.theta, i);
            self.conditioner.evaluate(simd, theta_i, units);
            self.conditioner.local_pullback(
                simd,
                theta_i,
                units,
                &cot,
                y[i],
                local,
                self.params_of_mut(grad, i),
            );
        }
    }
}
