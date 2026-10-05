//! The exact Gauss-Newton blocks of each variable's parameters, see
//! `notes/lm_derivatives.md` and `notes/lm_exact_blocks_identity.md`. For the
//! names, see the notation in [`super`].

use anyhow::{bail, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use fearless_simd::{dispatch, prelude::*};
use fearless_simd_macros::simd;
use rayon::prelude::*;

use super::conditioner::{DenseCurvature, Params, ParamsMut, UnitState};
#[cfg(doc)]
use super::selected_inverse::SelectedInverseLayout;
use super::simd::{axpy, axpy_lanes, dot};
use super::tiles::Linearization;
use super::FisherResiduals;
use crate::triangular::jet::{packed_len, JetArena};

/// Draws per matrix product when accumulating the exact blocks.
const BLOCK_DRAW_BATCH: usize = 32;

/// One exact Gauss-Newton sub-block: parameters `start..start + n` of the
/// flat parameter vector, `n = matrix.nrows()`.
pub(crate) struct GnBlock {
    pub(crate) start: usize,
    pub(crate) matrix: Mat<f64>,
}

/// Per-thread scratch for [`FisherResiduals::variable_blocks`]. The matrices
/// are column-major.
struct BlockScratch<S: Simd> {
    units: UnitState<S>,
    jets: JetArena<S>,
    /// The Jacobian rows of one tile, `n_theta` per column.
    rows: Vec<S::f64s>,
    edge_a: Vec<S::f64s>,
    curvature: DenseCurvature<S>,
    seeds: SeedProducts<S>,
}

/// The products [`FisherResiduals::block_rows`] shares between the seeds.
/// The matrices are column-major.
struct SeedProducts<S: Simd> {
    /// `C = dpi/dy_parents`, `(n_par, n_p)`.
    pi_jac: Vec<S::f64s>,
    /// `H_T[pi, pi] C` and `H_Lambda[pi, pi] C`, `(n_par, n_p)`.
    hess_t_jac: Vec<S::f64s>,
    hess_l_jac: Vec<S::f64s>,
    /// Every seed's `pi_bar`, `(n_par, n_col)`, and `W2^T pi_bar`,
    /// `(n_unit, n_col)`.
    pi_bar: Vec<S::f64s>,
    unit_bar: Vec<S::f64s>,
    /// `p_t` and `p_l` per unit, and one seed's `c_a t + c_l l`.
    unit_t: Vec<S::f64s>,
    unit_l: Vec<S::f64s>,
    edge_dir: Vec<S::f64s>,
}

/// One Jacobian row of variable `i`'s residuals in its parameters
/// `theta_i`, by the residual: `x_i` (only to fold into the others, see
/// [`FisherResiduals::block_rows`]), `i`'s own term of `q = J^T w - g +
/// grad_y log_det`, the term of parent `j`'s, and edge `j`'s score.
#[derive(Clone, Copy)]
enum Seed {
    X,
    Own,
    Edge(usize),
    Score(usize),
}

/// A [`Seed`]'s cotangent on the local outputs `(x_i, delta_i, mu_i)`, and
/// for an edge `j` the one-hot cotangents `c_a e_j` on `A[i, :]` and `c_l e_j`
/// on `L[i, :]`, as `(j, c_a, c_l)`.
struct SeedCotangent<S: Simd> {
    x: S::f64s,
    delta: S::f64s,
    mu: S::f64s,
    edge: Option<(usize, S::f64s, S::f64s)>,
}

impl Seed {
    /// The seed of row `col` of [`FisherResiduals::block_rows`].
    fn of_column(col: usize, n_parent: usize) -> Self {
        match col {
            0 => Seed::X,
            1 => Seed::Own,
            _ if col < 2 + n_parent => Seed::Edge(col - 2),
            _ => Seed::Score(col - 2 - n_parent),
        }
    }

    /// The cotangent, given `w_i`, `x_i` and `A[i, :]`.
    #[inline(always)]
    fn cotangent<S: Simd>(
        self,
        simd: S,
        w_i: S::f64s,
        x_i: S::f64s,
        edge_a: &[S::f64s],
    ) -> SeedCotangent<S> {
        let zero = S::f64s::splat(simd, 0.0);
        let one = S::f64s::splat(simd, 1.0);
        let (x, delta, mu, edge) = match self {
            Seed::X => (one, zero, zero, None),
            // q_own = -mu_i - delta_i w_i
            Seed::Own => (zero, -w_i, -one, None),
            // q_j = -L[i,j] - A[i,j] w_i
            Seed::Edge(j) => (zero, zero, zero, Some((j, -w_i, -one))),
            // score_j = L[i,j] - x_i A[i,j]
            Seed::Score(j) => (-edge_a[j], zero, zero, Some((j, -x_i, one))),
        };
        SeedCotangent { x, delta, mu, edge }
    }
}

impl FisherResiduals {
    /// `Sigma = (J^T J)^{-1}` on the diagonal and the edges, laid out by
    /// [`SelectedInverseLayout`], by Takahashi's recurrence:
    ///
    /// ```text
    /// Sigma[i, q] = -sum_p A[i, p] Sigma[p, q] / delta_i      (q in P(i))
    /// Sigma[i, i] = (1 / delta_i - sum_p A[i, p] Sigma[i, p]) / delta_i
    /// ```
    ///
    /// Row `i` reads only rows of earlier variables, and its own edges.
    #[simd]
    fn selected_inverse<S: Simd>(
        &self,
        simd: S,
        edge_a: &[S::f64s],
        delta: &[S::f64s],
        store: &mut [S::f64s],
    ) {
        let zero = S::f64s::splat(simd, 0.0);
        let one = S::f64s::splat(simd, 1.0);
        let layout = &self.inverse;
        for i in 0..self.n_var() {
            let edges = self.graph.edges(i);
            let (pairs, n_p) = (layout.parent_pairs(i), edges.len());
            for (b, e_q) in edges.clone().enumerate() {
                let row = &pairs[b * n_p..(b + 1) * n_p];
                let mut acc = zero;
                for (e_p, &slot) in edges.clone().zip(row) {
                    acc += edge_a[e_p] * store[slot as usize];
                }
                store[layout.edge(e_q)] = -acc / delta[i];
            }
            let mut acc = zero;
            for e in edges {
                acc += edge_a[e] * store[layout.edge(e)];
            }
            store[layout.diagonal(i)] = (one / delta[i] - acc) / delta[i];
        }
    }

    /// Exact Gauss-Newton blocks, `sum_draws (dr/dtheta_i)^T (dr/dtheta_i) /
    /// n_draw`, for each variable restricted to consecutive sub-blocks of at
    /// most `max_block_size` parameters of its slice. Ordered by variable,
    /// then position.
    pub(crate) fn gauss_newton_blocks(
        &self,
        lin: &Linearization,
        max_block_size: usize,
    ) -> Result<Vec<GnBlock>> {
        if max_block_size == 0 {
            bail!("max_block_size must be positive");
        }
        Ok(dispatch!(self.level, simd => self.gauss_newton_blocks_with(simd, lin, max_block_size)))
    }

    fn gauss_newton_blocks_with<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        max_block_size: usize,
    ) -> Vec<GnBlock> {
        let (n_var, n_edge) = (self.n_var(), self.n_edge());
        let zero = S::f64s::splat(simd, 0.0);
        let n_store = self.inverse.len();
        let mut stores = vec![zero; lin.data.n_tile() * n_store];
        stores.par_chunks_mut(n_store).enumerate().for_each_init(
            || (vec![zero; n_edge], vec![zero; n_var]),
            |(edge_a, delta), (tile, store)| {
                lin.edge_a.load(simd, tile, edge_a);
                lin.delta.load(simd, tile, delta);
                self.selected_inverse(simd, edge_a, delta, store);
            },
        );

        let per_var: Vec<Vec<GnBlock>> = (0..n_var)
            .into_par_iter()
            .map_init(
                || BlockScratch {
                    units: self.unit_state(simd),
                    jets: JetArena::new(simd),
                    rows: Vec::new(),
                    edge_a: Vec::with_capacity(self.graph.max_parent()),
                    curvature: DenseCurvature::new(simd, self.conditioner.n_par),
                    seeds: SeedProducts {
                        pi_jac: Vec::new(),
                        hess_t_jac: Vec::new(),
                        hess_l_jac: Vec::new(),
                        pi_bar: Vec::new(),
                        unit_bar: Vec::new(),
                        unit_t: vec![zero; self.conditioner.n_unit],
                        unit_l: vec![zero; self.conditioner.n_unit],
                        edge_dir: vec![zero; self.conditioner.n_par],
                    },
                },
                |b, i| self.variable_blocks(simd, lin, &stores, n_store, i, max_block_size, b),
            )
            .collect();
        per_var.into_iter().flatten().collect()
    }

    /// Variable `i`'s Jacobian rows for one tile, as columns of `b.rows`
    /// (`n_theta = |theta_i|` each): `a = dx_i/dtheta_i`, then the `n_s = 1 +
    /// n_p` rows of `B' = B + alpha a` (see [`Self::variable_blocks`]), then
    /// with regularization the `n_score = n_p` scores scaled by `sqrt(rho)`;
    /// `n_col` in all.
    #[simd]
    fn block_rows<S: Simd>(
        &self,
        simd: S,
        i: usize,
        theta: Params<f64>,
        tile: usize,
        lin: &Linearization,
        b: &mut BlockScratch<S>,
    ) {
        let shape = theta.shape;
        let n_theta = shape.size();
        let (n_p, n_s) = (shape.n_parent, shape.n_parent + 1);
        let n_score = if self.regularization.is_some() {
            n_p
        } else {
            0
        };
        let zero = S::f64s::splat(simd, 0.0);

        let BlockScratch {
            units,
            jets,
            rows,
            edge_a,
            curvature,
            seeds:
                SeedProducts {
                    pi_jac,
                    hess_t_jac,
                    hess_l_jac,
                    pi_bar,
                    unit_bar,
                    unit_t,
                    unit_l,
                    edge_dir,
                },
        } = b;
        let y = &lin.data.y;
        units.y_parents.clear();
        edge_a.clear();
        for (e, p) in self.graph.entries(i) {
            units.y_parents.push(y.value(simd, tile, p));
            edge_a.push(lin.edge_a.value(simd, tile, e));
        }
        let y_own = y.value(simd, tile, i);
        let w_i = lin.w.value(simd, tile, i);
        let x_i = lin.x.value(simd, tile, i);
        let delta_i = lin.delta.value(simd, tile, i);

        self.conditioner.evaluate(simd, theta, units);
        self.conditioner
            .dense_curvature(simd, y_own, &units.pi, jets, curvature);
        let DenseCurvature {
            grad_t,
            grad_l,
            hess_t,
            hess_l,
        } = &*curvature;
        let (n_unit, n_par, loc) = (shape.n_unit, shape.n_par, self.conditioner.location);
        let (t, l) = (&grad_t[1..], &grad_l[1..]);

        // Each Jacobian row is the local pullback (see `local_pullback`) of
        // one seed `(x_bar, delta_bar, mu_bar)` plus, for an edge `j`, the
        // one-hot cotangents `c_a e_j` on `A[i, :]` and `c_l e_j` on
        // `L[i, :]`. Through the units these collapse onto the columns of
        // `C = dpi/dy_parents = W2 diag(h') W1 + e_loc s^T`: `t_bar = c_a C
        // e_j` and `l_bar = c_l C e_j`. So all seeds together need the
        // products `C`, `H[pi, pi] C` and `W2^T pi_bar`.
        let seed = |col| Seed::of_column(col, n_p).cotangent(simd, w_i, x_i, edge_a);

        pi_jac.clear();
        pi_jac.resize(n_par * n_p, zero);
        for u in 0..n_unit {
            let (w1, w2) = (theta.w1(u), theta.w2(u));
            for (column, &w1) in pi_jac.chunks_exact_mut(n_par).zip(w1) {
                axpy(simd, units.h1[u] * w1, w2, column);
            }
        }
        for (column, &skip) in pi_jac.chunks_exact_mut(n_par).zip(theta.skip()) {
            column[loc] += skip;
        }

        // `H[pi, pi] C`, from the packed lower triangles.
        for jac in [&mut *hess_t_jac, &mut *hess_l_jac] {
            jac.clear();
            jac.resize(n_par * n_p, zero);
        }
        for r in 0..n_par {
            let row = packed_len(1 + r) + 1;
            for c in 0..=r {
                let (h_t, h_l) = (hess_t[row + c], hess_l[row + c]);
                for j in 0..n_p {
                    let (rj, cj) = (j * n_par + r, j * n_par + c);
                    hess_t_jac[rj] += h_t * pi_jac[cj];
                    hess_l_jac[rj] += h_l * pi_jac[cj];
                    if c < r {
                        hess_t_jac[cj] += h_t * pi_jac[rj];
                        hess_l_jac[cj] += h_l * pi_jac[rj];
                    }
                }
            }
        }

        // pi_bar = x_bar t + H_T[pi, z] (delta_bar, t_bar) + H_Lambda[pi, z]
        // (mu_bar, l_bar), with `z = (y_i, pi)`
        let n_col = 1 + n_s + n_score;
        pi_bar.clear();
        pi_bar.resize(n_col * n_par, zero);
        for (col, pi) in pi_bar.chunks_exact_mut(n_par).enumerate() {
            let cot = seed(col);
            for (m, value) in pi.iter_mut().enumerate() {
                let r = packed_len(1 + m);
                *value = cot.x * t[m] + cot.delta * hess_t[r] + cot.mu * hess_l[r];
            }
            if let Some((j, c_a, c_l)) = cot.edge {
                let column = j * n_par..(j + 1) * n_par;
                axpy_lanes(simd, c_a, &hess_t_jac[column.clone()], pi);
                axpy_lanes(simd, c_l, &hess_l_jac[column], pi);
            }
        }

        unit_bar.clear();
        unit_bar.resize(n_col * n_unit, zero);
        for u in 0..n_unit {
            let w2 = theta.w2(u);
            unit_t[u] = dot(simd, w2, t);
            unit_l[u] = dot(simd, w2, l);
            for (col, pi) in pi_bar.chunks_exact(n_par).enumerate() {
                unit_bar[col * n_unit + u] = dot(simd, w2, pi);
            }
        }

        rows.clear();
        rows.resize(n_col * n_theta, zero);
        let columns = rows
            .chunks_exact_mut(n_theta)
            .zip(pi_bar.chunks_exact(n_par));
        for (col, (out, pi)) in columns.enumerate() {
            let mut out = ParamsMut::new(out, shape);
            let edge = seed(col).edge;
            if let Some((_, c_a, c_l)) = edge {
                for ((dir, &t), &l) in edge_dir.iter_mut().zip(t).zip(l) {
                    *dir = c_a * t + c_l * l;
                }
            }
            for u in 0..n_unit {
                let (h, h1) = (units.h[u], units.h1[u]);
                let mut a_bar = h1 * unit_bar[col * n_unit + u];
                let out_w2 = out.w2(u);
                for (out, &pi) in out_w2.iter_mut().zip(pi) {
                    *out = h * pi;
                }
                // The edge cotangents' paths through `W1[u, j]`.
                let mut w1_direct = None;
                if let Some((j, c_a, c_l)) = edge {
                    let w1 = theta.w1(u)[j];
                    let p = c_a * unit_t[u] + c_l * unit_l[u];
                    a_bar += units.h2[u] * p * w1;
                    axpy_lanes(simd, h1 * w1, &edge_dir[..], out_w2);
                    w1_direct = Some((j, h1 * p));
                }
                let out_w1 = out.w1(u);
                for (out, &y) in out_w1.iter_mut().zip(&units.y_parents) {
                    *out = a_bar * y;
                }
                if let Some((j, value)) = w1_direct {
                    out_w1[j] += value;
                }
                *out.b1(u) = a_bar;
            }
            out.b2().copy_from_slice(pi);
            let out_skip = out.skip();
            for (out, &y) in out_skip.iter_mut().zip(&units.y_parents) {
                *out = pi[loc] * y;
            }
            if let Some((j, c_a, c_l)) = edge {
                out_skip[j] += c_a * t[loc] + c_l * l[loc];
            }
        }

        // Row `i` of `J` is `alpha = (delta_i, A[i, P(i)])` on `S_i`, so
        // `e_i = J^{-T} E_i alpha` and the direct part folds into the
        // solve: `dr/dtheta_i = J^{-T} E_i B'` with `B' = B + alpha a`.
        let (a, rest) = rows.split_at_mut(n_theta);
        axpy_lanes(simd, delta_i, a, &mut rest[..n_theta]);
        for j in 0..n_p {
            axpy_lanes(
                simd,
                edge_a[j],
                a,
                &mut rest[(1 + j) * n_theta..(2 + j) * n_theta],
            );
        }
        let reg = self.regularization.unwrap_or(0.0);
        for value in &mut rest[n_s * n_theta..] {
            *value *= reg;
        }
    }

    /// Variable `i`'s exact blocks (see `notes/lm_exact_blocks_identity.md`):
    ///
    /// ```text
    /// G = sum_draws B'^T K B' / n_draw   (+ the scores' Gram)
    /// ```
    ///
    /// where `B = dq/dtheta_i` for the local cotangent `q` on `S_i = {i} +
    /// P(i)` (size `n_s`), `alpha = (delta_i, A[i, P(i)])` is row `i` of `J`
    /// on `S_i`, `B' = B + alpha a`, and `K = Sigma[S_i, S_i]` from the
    /// selected inverse in `stores`, `n_store` slots per tile.
    #[allow(clippy::too_many_arguments)]
    fn variable_blocks<S: Simd>(
        &self,
        simd: S,
        lin: &Linearization,
        stores: &[S::f64s],
        n_store: usize,
        i: usize,
        max_block_size: usize,
        b: &mut BlockScratch<S>,
    ) -> Vec<GnBlock> {
        let shape = self.shape(i);
        let n_theta = shape.size();
        let n_p = shape.n_parent;
        let n_s = n_p + 1;
        let theta = self.params_of(&lin.theta, i);
        let n_score = if self.regularization.is_some() {
            n_p
        } else {
            0
        };

        let mut blocks: Vec<GnBlock> = (0..n_theta)
            .step_by(max_block_size)
            .map(|start| GnBlock {
                start: self.param_offset.start(i) + start,
                matrix: Mat::zeros(
                    max_block_size.min(n_theta - start),
                    max_block_size.min(n_theta - start),
                ),
            })
            .collect();

        // The slots of `K = Sigma` on `{i} + P(i)`, row-major.
        let mut k_slot = vec![self.inverse.diagonal(i); n_s * n_s];
        let pairs = self.inverse.parent_pairs(i);
        for (a, e) in self.graph.edges(i).enumerate() {
            k_slot[1 + a] = self.inverse.edge(e);
            k_slot[(1 + a) * n_s] = self.inverse.edge(e);
            for b in 0..n_p {
                k_slot[(1 + a) * n_s + 1 + b] = pairs[a * n_p + b] as usize;
            }
        }
        let mut k_mat = Mat::<f64>::zeros(n_s, n_s);

        // `G += P Q^T` over batches of draws, with the columns of `P`
        // (`p_mat`) each draw's `B'` and scaled score rows, and those of `Q`
        // (`q_mat`) the matching `K B'` and score rows: one matrix product per
        // batch and sub-block, instead of streaming every block once per row.
        let per_draw = n_s + n_score;
        let batch = BLOCK_DRAW_BATCH.min(lin.data.n_draw);
        let mut p_mat = Mat::<f64>::zeros(n_theta, batch * per_draw);
        let mut q_mat = Mat::<f64>::zeros(n_theta, batch * per_draw);
        let mut in_batch = 0;

        for tile in 0..lin.data.n_tile() {
            self.block_rows(simd, i, theta, tile, lin, b);
            let store = &stores[tile * n_store..(tile + 1) * n_store];
            for lane in 0..self.width {
                let draw = tile * self.width + lane;

                // `B'` and the scores of this draw, skipping `a`.
                let first = in_batch * per_draw;
                for c in 0..per_draw {
                    let rows = &b.rows[(1 + c) * n_theta..(2 + c) * n_theta];
                    for (out, row) in p_mat.col_as_slice_mut(first + c).iter_mut().zip(rows) {
                        *out = row[lane];
                    }
                }

                // K = Sigma on {i} + P(i), and K B' (as columns, B' K).
                for r in 0..n_s {
                    for c in 0..n_s {
                        k_mat[(r, c)] = store[k_slot[r * n_s + c]][lane];
                    }
                }
                matmul(
                    q_mat.as_mut().subcols_mut(first, n_s),
                    Accum::Replace,
                    p_mat.as_ref().subcols(first, n_s),
                    k_mat.as_ref(),
                    1.0,
                    Par::Seq,
                );
                q_mat
                    .as_mut()
                    .subcols_mut(first + n_s, n_score)
                    .copy_from(p_mat.as_ref().subcols(first + n_s, n_score));
                in_batch += 1;

                if in_batch == batch || draw + 1 == lin.data.n_draw {
                    let n_cols = in_batch * per_draw;
                    let local_start = self.param_offset.start(i);
                    for block in &mut blocks {
                        let start = block.start - local_start;
                        let size = block.matrix.nrows();
                        matmul(
                            block.matrix.as_mut(),
                            Accum::Add,
                            p_mat.as_ref().subrows(start, size).subcols(0, n_cols),
                            q_mat
                                .as_ref()
                                .subrows(start, size)
                                .subcols(0, n_cols)
                                .transpose(),
                            1.0,
                            Par::Seq,
                        );
                    }
                    in_batch = 0;
                }
            }
        }

        // `P Q^T` is symmetric only up to rounding.
        let half_inv_n = 0.5 / lin.data.n_draw as f64;
        for block in &mut blocks {
            let m = &mut block.matrix;
            for r in 0..m.nrows() {
                for c in 0..=r {
                    let value = (m[(r, c)] + m[(c, r)]) * half_inv_n;
                    m[(r, c)] = value;
                    m[(c, r)] = value;
                }
            }
        }
        blocks
    }
}
