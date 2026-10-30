//! The local kernels: one variable's [`Conditional`], and its
//! tangents and cotangents, for one tile of draws. For the names, see the
//! notation in [`super`].
//!
//! The conditioners give `pi` (`coefs`) and its derivatives in the parents
//! and the weights ([`crate::triangular::conditioner`]); the transformer, as
//! jets, gives `x`, `Lambda` and their derivatives in `(y_i, pi)`. Its
//! gradients in `pi`, `t` and `l`, are the stack of two cotangents every
//! edge entry is a parent pullback of: `A[i, :] = C^T t`, `L[i, :] = C^T l`.

use fearless_simd::prelude::*;
use fearless_simd_macros::simd;

use crate::triangular::conditional::Conditional;
use crate::triangular::conditioner::{Stack, StackMut, Workspace};
use crate::triangular::simd::dot_lanes;
use crate::triangular::transformer::jet::JetArena;
use crate::triangular::transformer::Transformer;

/// One variable's conditioners at its parents, see [`Conditional::evaluate`].
pub(super) struct LocalState<S: Simd> {
    /// The parents' values, filled before [`Conditional::evaluate`].
    pub(super) y_parents: Vec<S::f64s>,
    /// The transformer's coefficients.
    pub(super) pi: Vec<S::f64s>,
    /// The conditioners' states, after [`Conditional::evaluate`].
    pub(super) work: Workspace<S>,
}

/// Per-thread work space of the local kernels.
pub(super) struct LocalScratch<S: Simd> {
    pub(super) jets: JetArena<S>,
    /// Up to two jet directions in `z = (y_i, pi)`, `n_z` entries each.
    dirs: Vec<S::f64s>,
    /// `(t, l)`, and their tangents `(t_dot, l_dot)`.
    tl: Vec<S::f64s>,
    tl_dot: Vec<S::f64s>,
    /// `pi_dot` in the pushforward; `(t_bar, l_bar)` in the pullback.
    pi_dot: Vec<S::f64s>,
    pi_bar: Vec<S::f64s>,
    /// `(A, L)` of the variable's edges, or their tangents or cotangents.
    edges: Vec<S::f64s>,
}

/// A cotangent on one variable's local outputs `(x_i, delta_i, mu_i, A[i, :],
/// L[i, :])`: `x_bar`, `delta_bar`, `mu_bar`, `A_bar[i, :]` and
/// `L_bar[i, :]`, the last two one per parent.
pub(super) struct Cotangent<'a, S: Simd> {
    pub(super) x: S::f64s,
    pub(super) delta: S::f64s,
    pub(super) mu: S::f64s,
    pub(super) edge_a: &'a [S::f64s],
    pub(super) edge_l: &'a [S::f64s],
}

impl Conditional {
    pub(super) fn state<S: Simd>(&self, simd: S, max_parent: usize) -> LocalState<S> {
        LocalState {
            y_parents: Vec::with_capacity(max_parent),
            pi: vec![S::f64s::splat(simd, 0.0); self.n_coefs()],
            work: self.conditioners.workspace(simd, max_parent),
        }
    }

    pub(super) fn scratch<S: Simd>(&self, simd: S, max_parent: usize) -> LocalScratch<S> {
        let zero = S::f64s::splat(simd, 0.0);
        let (n_c, n_z) = (self.n_coefs(), 1 + self.n_coefs());
        LocalScratch {
            jets: JetArena::new(simd),
            dirs: vec![zero; 2 * n_z],
            tl: vec![zero; 2 * n_c],
            tl_dot: vec![zero; 2 * n_c],
            pi_dot: vec![zero; 2 * n_c],
            pi_bar: vec![zero; n_c],
            edges: Vec::with_capacity(2 * max_parent),
        }
    }

    /// One variable's conditioners at its parents, already in
    /// `st.y_parents`: the coefficients `pi` and every conditioner's state.
    /// Every local kernel reads its result.
    #[simd]
    pub(super) fn evaluate<S: Simd>(&self, simd: S, theta: &[f64], st: &mut LocalState<S>) {
        let LocalState {
            y_parents,
            pi,
            work,
        } = st;
        self.conditioners
            .forward_tile(simd, theta, y_parents, work, pi);
    }

    /// `t` and `l` from the transformer's gradients into `s.tl`.
    #[inline(always)]
    fn store_tl<S: Simd>(&self, grad_x: &[S::f64s], grad_log_det: &[S::f64s], tl: &mut [S::f64s]) {
        let n_c = self.n_coefs();
        tl[..n_c].copy_from_slice(&grad_x[1..]);
        tl[n_c..].copy_from_slice(&grad_log_det[1..]);
    }

    /// One variable's primal. Writes its edge values to `edge_a` and
    /// `edge_l` and returns `(x, delta, mu)`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_primal<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        st: &mut LocalState<S>,
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut LocalScratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let (x, log_det) =
            self.transformer
                .inverse_jet_tile(simd, &mut s.jets, &st.pi, y_own, &[], 0);
        let ws = &s.jets;
        self.store_tl::<S>(ws.grad(&x), ws.grad(&log_det), &mut s.tl);
        let out = (ws.value(&x), ws.grad(&x)[0], ws.grad(&log_det)[0]);

        let n_p = st.y_parents.len();
        s.edges.clear();
        s.edges.resize(2 * n_p, S::f64s::splat(simd, 0.0));
        if n_p > 0 {
            self.conditioners.pullback_parents_tile(
                simd,
                theta,
                &st.y_parents,
                &mut st.work,
                Stack::new(&s.tl, 2),
                StackMut::new(&mut s.edges, 2),
            );
        }
        edge_a.copy_from_slice(&s.edges[..n_p]);
        edge_l.copy_from_slice(&s.edges[n_p..]);
        out
    }

    /// One variable's tangents along `v`. Writes `A_dot[i, :]` and
    /// `L_dot[i, :]` to `edge_a` and `edge_l`; returns `(x_dot, delta_dot,
    /// mu_dot)`.
    ///
    /// `pi_dot` comes through the conditioners, `t_dot` and `l_dot` (and
    /// `delta_dot`, `mu_dot`) from one jet pass along `(0, pi_dot)`. The edge
    /// tangents are the tangent of `C^T (t, l)` along `v` and `(t_dot,
    /// l_dot)`.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_pushforward<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        v: &[f64],
        st: &mut LocalState<S>,
        y_own: S::f64s,
        edge_a: &mut [S::f64s],
        edge_l: &mut [S::f64s],
        s: &mut LocalScratch<S>,
    ) -> (S::f64s, S::f64s, S::f64s) {
        let (n_c, n_z) = (self.n_coefs(), 1 + self.n_coefs());
        let zero = S::f64s::splat(simd, 0.0);
        let pi_dot = &mut s.pi_dot[..n_c];
        pi_dot.fill(zero);
        let tangent = self.conditioners.pushforward_weights_tile(
            simd,
            theta,
            &st.y_parents,
            &mut st.work,
            v,
            pi_dot,
        );

        s.dirs[0] = zero;
        s.dirs[1..n_z].copy_from_slice(pi_dot);
        let (x, log_det) =
            self.transformer
                .inverse_jet_tile(simd, &mut s.jets, &st.pi, y_own, &s.dirs[..n_z], 1);
        let ws = &s.jets;
        self.store_tl::<S>(ws.grad(&x), ws.grad(&log_det), &mut s.tl);
        let (hess_x, hess_log_det) = (ws.hess_dir(&x, 0), ws.hess_dir(&log_det, 0));
        self.store_tl::<S>(hess_x, hess_log_det, &mut s.tl_dot);
        let out = (
            dot_lanes(simd, &s.tl[..n_c], &s.pi_dot[..n_c]),
            hess_x[0],
            hess_log_det[0],
        );

        let n_p = st.y_parents.len();
        s.edges.clear();
        s.edges.resize(2 * n_p, zero);
        self.conditioners.pullback_parents_tangent_tile(
            simd,
            theta,
            &st.y_parents,
            tangent,
            Stack::new(&s.tl, 2),
            Stack::new(&s.tl_dot, 2),
            StackMut::new(&mut s.edges, 2),
        );
        edge_a.copy_from_slice(&s.edges[..n_p]);
        edge_l.copy_from_slice(&s.edges[n_p..]);
        out
    }

    /// One variable's local pullback of `cot`, accumulated into `grad`, its
    /// slice of the gradient (one vector per weight).
    ///
    /// The edge cotangents reach `t` and `l` through `C`: `t_bar = C A_bar`,
    /// `l_bar = C L_bar`. One jet pass gives `pi_bar = x_bar t + H_T
    /// (delta_bar, t_bar) + H_Lambda (mu_bar, l_bar)` (restricted to `pi`),
    /// which goes back through the conditioners with the edge cotangents.
    #[simd]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn local_pullback<S: Simd>(
        &self,
        simd: S,
        theta: &[f64],
        st: &mut LocalState<S>,
        cot: &Cotangent<S>,
        y_own: S::f64s,
        s: &mut LocalScratch<S>,
        grad: &mut [S::f64s],
    ) {
        let (n_c, n_z) = (self.n_coefs(), 1 + self.n_coefs());
        let zero = S::f64s::splat(simd, 0.0);

        // The cotangents of `t` and `l`, from the edge cotangents.
        s.edges.clear();
        s.edges.extend_from_slice(cot.edge_a);
        s.edges.extend_from_slice(cot.edge_l);
        s.pi_dot.fill(zero);
        let pushed = self.conditioners.pushforward_parents_tile(
            simd,
            theta,
            &st.y_parents,
            &mut st.work,
            Stack::new(&s.edges, 2),
            StackMut::new(&mut s.pi_dot, 2),
        );

        // `T` along `(delta_bar, t_bar)` and `Lambda` along `(mu_bar,
        // l_bar)`: every second-order path into `pi` at once.
        s.dirs[0] = cot.delta;
        s.dirs[1..n_z].copy_from_slice(&s.pi_dot[..n_c]);
        s.dirs[n_z] = cot.mu;
        s.dirs[n_z + 1..2 * n_z].copy_from_slice(&s.pi_dot[n_c..]);
        let (x, log_det) =
            self.transformer
                .inverse_jet_tile(simd, &mut s.jets, &st.pi, y_own, &s.dirs, 2);
        let ws = &s.jets;
        self.store_tl::<S>(ws.grad(&x), ws.grad(&log_det), &mut s.tl);
        let (prod_t, prod_l) = (ws.hess_dir(&x, 0), ws.hess_dir(&log_det, 1));
        for m in 0..n_c {
            s.pi_bar[m] = cot.x * s.tl[m] + prod_t[1 + m] + prod_l[1 + m];
        }

        self.conditioners.pullback_weights_tile(
            simd,
            theta,
            &st.y_parents,
            pushed,
            &s.pi_bar,
            Stack::new(&s.tl, 2),
            grad,
        );
    }
}
