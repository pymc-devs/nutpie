//! Residuals of the Levenberg-Marquardt fit of a `SparseTriangularMap`, and
//! the derivatives `lmopt` needs from them: the pushforward `J v`, the
//! pullback `J^T r`, and the exact Gauss-Newton blocks. The derivation and the
//! notation are in `notes/lm_derivatives.md`.
//!
//! Only the conditioners are fitted. The map's inputs `y` and the gradients
//! `g` are fixed data, already mapped through the frozen affine and
//! permutation, so the residual of one draw is
//!
//! ```text
//! r = (x + w) / sqrt(n_draw),   J^T w = g - grad_y log_det,
//! ```
//!
//! followed, with a Fisher regularization `rho`, by one parent score per edge,
//! `sqrt(rho) (L[i,j] - x_i A[i,j]) / sqrt(n_draw)`.
//!
//! Supported conditioners: one hidden layer with softplus, and the linear
//! location skip. The transformer is a `Contract2` chain, evaluated in the
//! density direction.
//!
//! Parameters are laid out per variable, with no bucketing or padding.
//! Variable `i`'s slice holds, for each hidden unit `u`, the unit's input
//! weights `W1[u, :]`, its bias `b1[u]` and its output weights `W2[:, u]`,
//! followed by `b2` and the skip weights `s`. Every kernel walks the
//! conditioner unit by unit, so this keeps each unit's weights contiguous.

use anyhow::{bail, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::triangular::{Contract2Spec, Param};

/// Capacity of a [`Jet`]: the transformer input `y` and up to 15 transformer
/// parameters (three `Contract2` layers).
const N: usize = 16;
/// Number of directions a [`Jet`] carries second derivatives along.
const K: usize = 2;

/// A scalar function of `z = (y, pi)`, with its gradient and the products of
/// its Hessian with `K` fixed directions: `e[k] = g . d_k`, `h[k] = H d_k`.
///
/// Forward-over-forward, but sharing the value and gradient across the
/// directions, so every operation is O(N) instead of the O(N^2) a full
/// Hessian would cost.
#[derive(Clone, Copy)]
struct Jet {
    v: f64,
    g: [f64; N],
    e: [f64; K],
    h: [[f64; N]; K],
}

impl Jet {
    #[inline(always)]
    fn constant(v: f64) -> Self {
        Self {
            v,
            g: [0.0; N],
            e: [0.0; K],
            h: [[0.0; N]; K],
        }
    }

    #[inline(always)]
    fn variable(v: f64, index: usize, dirs: &[[f64; N]; K]) -> Self {
        let mut out = Self::constant(v);
        out.g[index] = 1.0;
        for k in 0..K {
            out.e[k] = dirs[k][index];
        }
        out
    }

    /// `f(self)`, given `f`, `f'` and `f''` at `self.v`.
    #[inline(always)]
    fn chain(self, f0: f64, f1: f64, f2: f64) -> Self {
        let mut out = Self::constant(f0);
        for n in 0..N {
            out.g[n] = f1 * self.g[n];
        }
        for k in 0..K {
            out.e[k] = f1 * self.e[k];
            let curvature = f2 * self.e[k];
            for n in 0..N {
                out.h[k][n] = f1 * self.h[k][n] + curvature * self.g[n];
            }
        }
        out
    }

    #[inline(always)]
    fn asinh(self) -> Self {
        let a = self.v;
        let root = a.hypot(1.0);
        self.chain(a.asinh(), 1.0 / root, -a / (root * root * root))
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        let s = self.v.sinh();
        self.chain(s, self.v.cosh(), s)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e, e)
    }

    #[inline(always)]
    fn ln_1p(self) -> Self {
        let inv = 1.0 / (1.0 + self.v);
        self.chain(self.v.ln_1p(), inv, -inv * inv)
    }

    /// `log(cosh(v))`, in `_log_cosh`'s stable form.
    #[inline(always)]
    fn log_cosh(self) -> Self {
        let a = self.v.abs();
        let t = self.v.tanh();
        self.chain(
            a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2,
            t,
            1.0 - t * t,
        )
    }

    #[inline(always)]
    fn sigmoid(self) -> Self {
        let s = 1.0 / (1.0 + (-self.v).exp());
        let ds = s * (1.0 - s);
        self.chain(s, ds, ds * (1.0 - 2.0 * s))
    }

    #[inline(always)]
    fn square(self) -> Self {
        self * self
    }
}

impl std::ops::Add for Jet {
    type Output = Jet;
    #[inline(always)]
    fn add(mut self, other: Jet) -> Jet {
        self.v += other.v;
        for n in 0..N {
            self.g[n] += other.g[n];
        }
        for k in 0..K {
            self.e[k] += other.e[k];
            for n in 0..N {
                self.h[k][n] += other.h[k][n];
            }
        }
        self
    }
}

impl std::ops::Neg for Jet {
    type Output = Jet;
    #[inline(always)]
    fn neg(self) -> Jet {
        self * -1.0
    }
}

impl std::ops::Sub for Jet {
    type Output = Jet;
    #[inline(always)]
    fn sub(self, other: Jet) -> Jet {
        self + (-other)
    }
}

impl std::ops::Mul for Jet {
    type Output = Jet;
    #[inline(always)]
    fn mul(self, other: Jet) -> Jet {
        let (a, b) = (self, other);
        let mut out = Jet::constant(a.v * b.v);
        for n in 0..N {
            out.g[n] = a.v * b.g[n] + b.v * a.g[n];
        }
        for k in 0..K {
            out.e[k] = a.v * b.e[k] + b.v * a.e[k];
            for n in 0..N {
                out.h[k][n] = a.v * b.h[k][n] + b.v * a.h[k][n] + a.e[k] * b.g[n] + b.e[k] * a.g[n];
            }
        }
        out
    }
}

impl std::ops::Mul<f64> for Jet {
    type Output = Jet;
    #[inline(always)]
    fn mul(mut self, c: f64) -> Jet {
        self.v *= c;
        for n in 0..N {
            self.g[n] *= c;
        }
        for k in 0..K {
            self.e[k] *= c;
            for n in 0..N {
                self.h[k][n] *= c;
            }
        }
        self
    }
}

impl std::ops::Add<f64> for Jet {
    type Output = Jet;
    #[inline(always)]
    fn add(mut self, c: f64) -> Jet {
        self.v += c;
        self
    }
}

/// `_bounded_log_gamma`'s constants for one pair of bounds.
#[derive(Debug, Clone, Copy)]
struct GammaBound {
    low: f64,
    width: f64,
    slope: f64,
    offset: f64,
}

#[derive(Debug, Clone, Copy)]
struct Layer {
    alpha: Option<Param>,
    beta: Option<Param>,
    sigma: Option<Param>,
    mu: Option<Param>,
    nu: Option<Param>,
    bound: Option<GammaBound>,
}

impl Layer {
    fn new(spec: Contract2Spec) -> Result<Self> {
        let bound = match spec.log_gamma_bounds {
            None => None,
            Some((low, high)) => {
                if !(low < 0.0 && 0.0 < high) {
                    bail!("log_gamma_bounds must satisfy low < 0 < high, got ({low}, {high})");
                }
                let width = high - low;
                let at_zero = -low / width;
                Some(GammaBound {
                    low,
                    width,
                    slope: width / (-low * high),
                    offset: (at_zero / (1.0 - at_zero)).ln(),
                })
            }
        };
        Ok(Self {
            alpha: spec.alpha,
            beta: spec.beta,
            sigma: spec.sigma,
            mu: spec.mu,
            nu: spec.nu,
            bound,
        })
    }

    fn params(&self) -> impl Iterator<Item = Param> {
        [self.alpha, self.beta, self.sigma, self.mu, self.nu]
            .into_iter()
            .flatten()
    }
}

/// `T(y; pi)` and `Lambda(y; pi) = log |dT/dy|` of the inverted `Contract2`
/// chain (`Contract2.inverse_and_log_det`, last layer first), as jets in
/// `z = (y, pi)` along the directions `dirs`.
fn transformer(layers: &[Layer], y: f64, pi: &[f64], dirs: &[[f64; N]; K]) -> (Jet, Jet) {
    let field = |param: Option<Param>| {
        param.map(|p| Jet::variable(pi[p.index] + p.offset, 1 + p.index, dirs))
    };

    let mut x = Jet::variable(y, 0, dirs);
    let mut log_det = Jet::constant(0.0);
    for layer in layers.iter().rev() {
        let log_gamma = field(layer.alpha).map(|alpha| {
            let log_gamma = alpha.asinh();
            match layer.bound {
                None => log_gamma,
                Some(b) => (log_gamma * b.slope + b.offset).sigmoid() * b.width + b.low,
            }
        });
        let log_delta = field(layer.beta).map(Jet::asinh);
        let log_sigma = field(layer.sigma).map(Jet::asinh);

        let centred = match field(layer.mu) {
            Some(mu) => x - mu,
            None => x,
        };
        let log_scale = match (log_gamma, log_sigma) {
            (Some(g), Some(s)) => Some(g - s),
            (Some(g), None) => Some(g),
            (None, Some(s)) => Some(-s),
            (None, None) => None,
        };
        let half_a = match log_scale {
            Some(scale) => scale.exp() * centred * 0.5,
            None => centred * 0.5,
        };
        let arg = half_a.asinh();
        let shifted = match log_delta {
            Some(delta) => arg - delta * 2.0,
            None => arg,
        };
        let u = match log_gamma {
            Some(g) => shifted * (-g).exp(),
            None => shifted,
        };

        let mut out = u.sinh() * 2.0;
        if let Some(nu) = field(layer.nu) {
            out = out + nu;
        }
        let mut ld = u.log_cosh() - half_a.square().ln_1p() * 0.5;
        if let Some(s) = log_sigma {
            ld = ld - s;
        }
        x = out;
        log_det = log_det + ld;
    }
    (x, log_det)
}

/// Softplus and its first two derivatives.
#[inline(always)]
fn softplus(a: f64) -> (f64, f64, f64) {
    let e = (-a.abs()).exp();
    let sigmoid = if a >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    };
    (e.ln_1p() + a.max(0.0), sigmoid, sigmoid * (1.0 - sigmoid))
}

#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[inline(always)]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// Where one variable's conditioner parameters sit in its slice.
#[derive(Clone, Copy)]
struct Shape {
    n_parent: usize,
    n_unit: usize,
    n_par: usize,
}

impl Shape {
    #[inline(always)]
    fn stride(&self) -> usize {
        self.n_parent + 1 + self.n_par
    }

    fn size(&self) -> usize {
        self.n_unit * self.stride() + self.n_par + self.n_parent
    }

    #[inline(always)]
    fn w1(&self, u: usize) -> std::ops::Range<usize> {
        let start = u * self.stride();
        start..start + self.n_parent
    }

    #[inline(always)]
    fn b1(&self, u: usize) -> usize {
        u * self.stride() + self.n_parent
    }

    #[inline(always)]
    fn w2(&self, u: usize) -> std::ops::Range<usize> {
        let start = u * self.stride() + self.n_parent + 1;
        start..start + self.n_par
    }

    #[inline(always)]
    fn b2(&self) -> std::ops::Range<usize> {
        let start = self.n_unit * self.stride();
        start..start + self.n_par
    }

    #[inline(always)]
    fn skip(&self) -> std::ops::Range<usize> {
        let start = self.n_unit * self.stride() + self.n_par;
        start..start + self.n_parent
    }
}

/// A cotangent on one variable's local outputs `(x_i, delta_i, mu_i, A[i, :],
/// L[i, :])`.
struct Cotangent<'a> {
    x: f64,
    delta: f64,
    mu: f64,
    edge_a: &'a [f64],
    edge_l: &'a [f64],
}

/// Per-thread scratch for the local kernels.
#[derive(Default)]
struct Scratch {
    y_parents: Vec<f64>,
    h: Vec<f64>,
    h1: Vec<f64>,
    h2: Vec<f64>,
    unit_a: Vec<f64>,
    unit_b: Vec<f64>,
    pi: Vec<f64>,
    pi_dot: Vec<f64>,
    t: Vec<f64>,
    l: Vec<f64>,
    t_bar: Vec<f64>,
    l_bar: Vec<f64>,
    pi_bar: Vec<f64>,
}

impl Scratch {
    fn new(n_unit: usize, n_par: usize, max_parent: usize) -> Self {
        Self {
            y_parents: vec![0.0; max_parent],
            h: vec![0.0; n_unit],
            h1: vec![0.0; n_unit],
            h2: vec![0.0; n_unit],
            unit_a: vec![0.0; n_unit],
            unit_b: vec![0.0; n_unit],
            pi: vec![0.0; n_par],
            pi_dot: vec![0.0; n_par],
            t: vec![0.0; n_par],
            l: vec![0.0; n_par],
            t_bar: vec![0.0; n_par],
            l_bar: vec![0.0; n_par],
            pi_bar: vec![0.0; n_par],
        }
    }
}

/// Full Hessians of `T` and `Lambda` in `z = (y, pi)`, for many cotangents
/// against the same point.
struct DenseTransformer {
    t: [f64; N],
    l: [f64; N],
    hess_t: [[f64; N]; N],
    hess_l: [[f64; N]; N],
}

/// One draw's quantities from the primal, which every later operation reads.
struct DrawTape<'a> {
    /// Pre-activations, `(n_var, n_unit)`.
    a: &'a [f64],
    edge_a: &'a [f64],
    delta: &'a [f64],
    x: &'a [f64],
    w: &'a [f64],
}

struct DrawTapeMut<'a> {
    a: &'a mut [f64],
    edge_a: &'a mut [f64],
    edge_l: &'a mut [f64],
    delta: &'a mut [f64],
    x: &'a mut [f64],
    w: &'a mut [f64],
}

#[derive(Default)]
struct Tape {
    theta: Vec<f64>,
    a: Vec<f64>,
    edge_a: Vec<f64>,
    edge_l: Vec<f64>,
    delta: Vec<f64>,
    x: Vec<f64>,
    w: Vec<f64>,
}

/// Splits `values` into `n` consecutive chunks of `size`, which may be zero.
fn chunks(values: &mut [f64], size: usize, n: usize) -> Vec<&mut [f64]> {
    let mut out = Vec::with_capacity(n);
    let mut rest = values;
    for _ in 0..n {
        let (head, tail) = rest.split_at_mut(size);
        out.push(head);
        rest = tail;
    }
    out
}

/// Per-draw scratch for the global sweeps.
struct DrawScratch {
    local: Scratch,
    by_var: [Vec<f64>; 4],
    by_edge: [Vec<f64>; 2],
    residual: Vec<f64>,
}

impl DrawScratch {
    fn new(problem: &FisherResiduals) -> Self {
        let n_var = problem.n_var;
        let n_edge = problem.n_edge();
        Self {
            local: problem.scratch(),
            by_var: std::array::from_fn(|_| vec![0.0; n_var]),
            by_edge: std::array::from_fn(|_| vec![0.0; n_edge]),
            residual: vec![0.0; problem.n_residuals()],
        }
    }
}

pub struct FisherResiduals {
    n_var: usize,
    n_unit: usize,
    n_par: usize,
    location: usize,
    parent_indptr: Vec<usize>,
    parent_index: Vec<usize>,
    max_parent: usize,
    param_offset: Vec<usize>,
    layers: Vec<Layer>,
    /// `sqrt(fisher_regularization)`, if regularized.
    regularization: Option<f64>,
    cholesky_jitter: Option<f64>,
    /// The parent sets closed under elimination, on which the selected
    /// inverse is stored. Equal to the parents for a chordal pattern.
    filled_indptr: Vec<usize>,
    filled_index: Vec<usize>,
    n_draw: usize,
    y: Vec<f64>,
    g: Vec<f64>,
    tape: Option<Tape>,
}

impl FisherResiduals {
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: Vec<usize>,
        parent_index: Vec<usize>,
        n_unit: usize,
        n_par: usize,
        location: usize,
        specs: Vec<Contract2Spec>,
        fisher_regularization: Option<f64>,
        cholesky_jitter: Option<f64>,
    ) -> Result<Self> {
        let Some(n_var) = parent_indptr.len().checked_sub(1) else {
            bail!("parent_indptr must not be empty");
        };
        if parent_indptr[0] != 0
            || parent_indptr.windows(2).any(|w| w[0] > w[1])
            || parent_indptr[n_var] != parent_index.len()
        {
            bail!("parent_indptr is not a valid CSR index pointer");
        }
        for i in 0..n_var {
            for &p in &parent_index[parent_indptr[i]..parent_indptr[i + 1]] {
                if p >= i {
                    bail!("variable {i} has parent {p}, which does not precede it");
                }
            }
        }
        if n_par + 1 > N {
            bail!(
                "the transformer has {n_par} parameters; at most {} are supported",
                N - 1
            );
        }
        if location >= n_par {
            bail!("location index {location} out of range for {n_par} parameters");
        }
        let layers = specs
            .into_iter()
            .map(Layer::new)
            .collect::<Result<Vec<_>>>()?;
        for param in layers.iter().flat_map(Layer::params) {
            if param.index >= n_par {
                bail!("transformer parameter index {} out of range", param.index);
            }
        }
        if let Some(rho) = fisher_regularization {
            if rho.is_nan() || rho < 0.0 {
                bail!("fisher_regularization must be non-negative, got {rho}");
            }
        }

        let max_parent = (0..n_var)
            .map(|i| parent_indptr[i + 1] - parent_indptr[i])
            .max()
            .unwrap_or(0);
        let mut param_offset = vec![0usize; n_var + 1];
        for i in 0..n_var {
            let shape = Shape {
                n_parent: parent_indptr[i + 1] - parent_indptr[i],
                n_unit,
                n_par,
            };
            param_offset[i + 1] = param_offset[i] + shape.size();
        }

        // Symbolic fill, as `_build_selected_inverse`: each `p` in `P*(i)`
        // must see the earlier part of `P*(i)`. Walking backwards finalizes
        // `P*(i)` before it propagates.
        let mut filled: Vec<std::collections::BTreeSet<usize>> = (0..n_var)
            .map(|i| {
                parent_index[parent_indptr[i]..parent_indptr[i + 1]]
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        for i in (0..n_var).rev() {
            let members: Vec<usize> = filled[i].iter().copied().collect();
            for (k, &p) in members.iter().enumerate() {
                filled[p].extend(members[..k].iter().copied());
            }
        }
        let mut filled_indptr = vec![0usize; n_var + 1];
        let mut filled_index = Vec::new();
        for i in 0..n_var {
            filled_index.extend(filled[i].iter().copied());
            filled_indptr[i + 1] = filled_index.len();
        }

        Ok(Self {
            n_var,
            n_unit,
            n_par,
            location,
            parent_indptr,
            parent_index,
            max_parent,
            param_offset,
            layers,
            regularization: fisher_regularization.map(f64::sqrt),
            cholesky_jitter,
            filled_indptr,
            filled_index,
            n_draw: 0,
            y: Vec::new(),
            g: Vec::new(),
            tape: None,
        })
    }

    fn n_edge(&self) -> usize {
        self.parent_index.len()
    }

    fn n_params(&self) -> usize {
        self.param_offset[self.n_var]
    }

    fn n_residuals(&self) -> usize {
        self.n_var
            + if self.regularization.is_some() {
                self.n_edge()
            } else {
                0
            }
    }

    fn scale(&self) -> f64 {
        1.0 / (self.n_draw as f64).sqrt()
    }

    fn scratch(&self) -> Scratch {
        Scratch::new(self.n_unit, self.n_par, self.max_parent)
    }

    fn shape(&self, i: usize) -> Shape {
        Shape {
            n_parent: self.parent_indptr[i + 1] - self.parent_indptr[i],
            n_unit: self.n_unit,
            n_par: self.n_par,
        }
    }

    fn edges(&self, i: usize) -> std::ops::Range<usize> {
        self.parent_indptr[i]..self.parent_indptr[i + 1]
    }

    fn params(&self, i: usize) -> std::ops::Range<usize> {
        self.param_offset[i]..self.param_offset[i + 1]
    }

    fn set_data(&mut self, y: Vec<f64>, g: Vec<f64>) -> Result<()> {
        if self.n_var == 0 || !y.len().is_multiple_of(self.n_var) || y.len() != g.len() {
            bail!("y and g must both have shape (n_draw, {})", self.n_var);
        }
        self.n_draw = y.len() / self.n_var;
        if self.n_draw == 0 {
            bail!("need at least one draw");
        }
        self.y = y;
        self.g = g;
        self.tape = None;
        Ok(())
    }

    fn tape(&self) -> Result<&Tape> {
        match &self.tape {
            Some(tape) => Ok(tape),
            None => bail!("no tape: call `residuals(theta, record=True)` first"),
        }
    }

    fn draw_tape<'a>(&self, tape: &'a Tape, draw: usize) -> DrawTape<'a> {
        let (n_var, n_edge, n_unit) = (self.n_var, self.n_edge(), self.n_unit);
        DrawTape {
            a: &tape.a[draw * n_var * n_unit..(draw + 1) * n_var * n_unit],
            edge_a: &tape.edge_a[draw * n_edge..(draw + 1) * n_edge],
            delta: &tape.delta[draw * n_var..(draw + 1) * n_var],
            x: &tape.x[draw * n_var..(draw + 1) * n_var],
            w: &tape.w[draw * n_var..(draw + 1) * n_var],
        }
    }

    fn gather_parents(&self, i: usize, y: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(self.parent_index[self.edges(i)].iter().map(|&p| y[p]));
    }

    // ------------------------------------------------------------ local kernels

    /// `h`, `h'`, `h''` at the pre-activations `a`, and `pi`.
    fn prepare(&self, shape: Shape, theta: &[f64], a: &[f64], s: &mut Scratch) {
        s.pi.copy_from_slice(&theta[shape.b2()]);
        for u in 0..shape.n_unit {
            let (h, h1, h2) = softplus(a[u]);
            s.h[u] = h;
            s.h1[u] = h1;
            s.h2[u] = h2;
            axpy(h, &theta[shape.w2(u)], &mut s.pi);
        }
        s.pi[self.location] += dot(&theta[shape.skip()], &s.y_parents);
    }

    fn dense_transformer(&self, y: f64, pi: &[f64]) -> DenseTransformer {
        let n_z = 1 + self.n_par;
        let mut out = DenseTransformer {
            t: [0.0; N],
            l: [0.0; N],
            hess_t: [[0.0; N]; N],
            hess_l: [[0.0; N]; N],
        };
        for first in (0..n_z).step_by(K) {
            let mut dirs = [[0.0; N]; K];
            for k in 0..K {
                if first + k < n_z {
                    dirs[k][first + k] = 1.0;
                }
            }
            let (x, log_det) = transformer(&self.layers, y, pi, &dirs);
            out.t = x.g;
            out.l = log_det.g;
            for k in 0..K {
                if first + k < n_z {
                    out.hess_t[first + k] = x.h[k];
                    out.hess_l[first + k] = log_det.h[k];
                }
            }
        }
        out
    }

    /// Variable `i`'s primal. Writes its pre-activations to `a` and its edge
    /// values to `edge_a` and `edge_l`; returns `(x, delta, mu)`.
    #[allow(clippy::too_many_arguments)]
    fn local_primal(
        &self,
        i: usize,
        theta: &[f64],
        y_own: f64,
        a: &mut [f64],
        edge_a: &mut [f64],
        edge_l: &mut [f64],
        s: &mut Scratch,
    ) -> (f64, f64, f64) {
        let shape = self.shape(i);
        for u in 0..shape.n_unit {
            a[u] = dot(&theta[shape.w1(u)], &s.y_parents) + theta[shape.b1(u)];
        }
        self.prepare(shape, theta, a, s);
        let (x, log_det) = transformer(&self.layers, y_own, &s.pi, &[[0.0; N]; K]);
        let t = &x.g[1..1 + self.n_par];
        let l = &log_det.g[1..1 + self.n_par];

        let skip = &theta[shape.skip()];
        for j in 0..shape.n_parent {
            edge_a[j] = t[self.location] * skip[j];
            edge_l[j] = l[self.location] * skip[j];
        }
        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let w1 = &theta[shape.w1(u)];
            axpy(dot(t, w2) * s.h1[u], w1, edge_a);
            axpy(dot(l, w2) * s.h1[u], w1, edge_l);
        }
        (x.v, x.g[0], log_det.g[0])
    }

    /// Variable `i`'s tangents along `v` (its slice). Writes `A`'s and `L`'s
    /// to `edge_a` and `edge_l`; returns those of `(x, delta, mu)`.
    #[allow(clippy::too_many_arguments)]
    fn local_pushforward(
        &self,
        i: usize,
        theta: &[f64],
        v: &[f64],
        y_own: f64,
        a: &[f64],
        edge_a: &mut [f64],
        edge_l: &mut [f64],
        s: &mut Scratch,
    ) -> (f64, f64, f64) {
        let shape = self.shape(i);
        self.prepare(shape, theta, a, s);

        s.pi_dot.copy_from_slice(&v[shape.b2()]);
        for u in 0..shape.n_unit {
            let a_dot = dot(&v[shape.w1(u)], &s.y_parents) + v[shape.b1(u)];
            s.unit_a[u] = a_dot;
            axpy(s.h[u], &v[shape.w2(u)], &mut s.pi_dot);
            axpy(s.h1[u] * a_dot, &theta[shape.w2(u)], &mut s.pi_dot);
        }
        s.pi_dot[self.location] += dot(&v[shape.skip()], &s.y_parents);

        let mut dirs = [[0.0; N]; K];
        dirs[0][1..1 + self.n_par].copy_from_slice(&s.pi_dot);
        let (x, log_det) = transformer(&self.layers, y_own, &s.pi, &dirs);
        let n_par = self.n_par;
        let t = &x.g[1..1 + n_par];
        let l = &log_det.g[1..1 + n_par];
        let t_dot = &x.h[0][1..1 + n_par];
        let l_dot = &log_det.h[0][1..1 + n_par];

        let skip = &theta[shape.skip()];
        let skip_dot = &v[shape.skip()];
        let loc = self.location;
        for j in 0..shape.n_parent {
            edge_a[j] = t_dot[loc] * skip[j] + t[loc] * skip_dot[j];
            edge_l[j] = l_dot[loc] * skip[j] + l[loc] * skip_dot[j];
        }
        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let w2_dot = &v[shape.w2(u)];
            let (p_t, p_l) = (dot(t, w2), dot(l, w2));
            let curvature = s.h2[u] * s.unit_a[u];
            let alpha_t = (dot(t_dot, w2) + dot(t, w2_dot)) * s.h1[u] + p_t * curvature;
            let alpha_l = (dot(l_dot, w2) + dot(l, w2_dot)) * s.h1[u] + p_l * curvature;
            let w1 = &theta[shape.w1(u)];
            let w1_dot = &v[shape.w1(u)];
            axpy(alpha_t, w1, edge_a);
            axpy(p_t * s.h1[u], w1_dot, edge_a);
            axpy(alpha_l, w1, edge_l);
            axpy(p_l * s.h1[u], w1_dot, edge_l);
        }
        (x.e[0], x.h[0][0], log_det.h[0][0])
    }

    /// First half of the local pullback: the cotangents of `t` and `l`, from
    /// the edge cotangents. Leaves `zeta` for `A` and `L` in `unit_a` and
    /// `unit_b`. Needs `prepare`.
    fn pullback_directions(&self, shape: Shape, theta: &[f64], cot: &Cotangent, s: &mut Scratch) {
        let skip = &theta[shape.skip()];
        s.t_bar.fill(0.0);
        s.l_bar.fill(0.0);
        for u in 0..shape.n_unit {
            let w1 = &theta[shape.w1(u)];
            let zeta_a = dot(cot.edge_a, w1);
            let zeta_l = dot(cot.edge_l, w1);
            s.unit_a[u] = zeta_a;
            s.unit_b[u] = zeta_l;
            let w2 = &theta[shape.w2(u)];
            axpy(s.h1[u] * zeta_a, w2, &mut s.t_bar);
            axpy(s.h1[u] * zeta_l, w2, &mut s.l_bar);
        }
        s.t_bar[self.location] += dot(cot.edge_a, skip);
        s.l_bar[self.location] += dot(cot.edge_l, skip);
    }

    /// Second half of the local pullback, given `t`, `l` and `pi_bar` (in
    /// `s.t`, `s.l`, `s.pi_bar`). Accumulates into `grad`, the variable's
    /// slice.
    fn pullback_finish(
        &self,
        shape: Shape,
        theta: &[f64],
        cot: &Cotangent,
        s: &Scratch,
        grad: &mut [f64],
    ) {
        let loc = self.location;
        for u in 0..shape.n_unit {
            let w2 = &theta[shape.w2(u)];
            let (p_t, p_l) = (dot(&s.t, w2), dot(&s.l, w2));
            let (zeta_a, zeta_l) = (s.unit_a[u], s.unit_b[u]);
            let a_bar = s.h2[u] * (p_t * zeta_a + p_l * zeta_l) + s.h1[u] * dot(&s.pi_bar, w2);

            let g_w2 = &mut grad[shape.w2(u)];
            axpy(s.h1[u] * zeta_a, &s.t, g_w2);
            axpy(s.h1[u] * zeta_l, &s.l, g_w2);
            axpy(s.h[u], &s.pi_bar, g_w2);

            let g_w1 = &mut grad[shape.w1(u)];
            axpy(s.h1[u] * p_t, cot.edge_a, g_w1);
            axpy(s.h1[u] * p_l, cot.edge_l, g_w1);
            axpy(a_bar, &s.y_parents, g_w1);
            grad[shape.b1(u)] += a_bar;
        }
        for (g, p) in grad[shape.b2()].iter_mut().zip(&s.pi_bar) {
            *g += p;
        }
        let g_skip = &mut grad[shape.skip()];
        axpy(s.t[loc], cot.edge_a, g_skip);
        axpy(s.l[loc], cot.edge_l, g_skip);
        axpy(s.pi_bar[loc], &s.y_parents, g_skip);
    }

    /// The whole local pullback, with one transformer pass along the two
    /// directions the cotangent asks for.
    #[allow(clippy::too_many_arguments)]
    fn local_pullback(
        &self,
        i: usize,
        theta: &[f64],
        y_own: f64,
        a: &[f64],
        cot: &Cotangent,
        s: &mut Scratch,
        grad: &mut [f64],
    ) {
        let shape = self.shape(i);
        let n_par = self.n_par;
        self.prepare(shape, theta, a, s);
        self.pullback_directions(shape, theta, cot, s);

        // `T` is seeded along `(delta_bar, t_bar)` and `Lambda` along
        // `(mu_bar, l_bar)`, so `H_T d_T + H_Lambda d_Lambda` is every second
        // order path into `pi` at once.
        let mut dirs = [[0.0; N]; K];
        dirs[0][0] = cot.delta;
        dirs[0][1..1 + n_par].copy_from_slice(&s.t_bar);
        dirs[1][0] = cot.mu;
        dirs[1][1..1 + n_par].copy_from_slice(&s.l_bar);
        let (x, log_det) = transformer(&self.layers, y_own, &s.pi, &dirs);
        s.t.copy_from_slice(&x.g[1..1 + n_par]);
        s.l.copy_from_slice(&log_det.g[1..1 + n_par]);
        for m in 0..n_par {
            s.pi_bar[m] = cot.x * s.t[m] + x.h[0][1 + m] + log_det.h[1][1 + m];
        }
        self.pullback_finish(shape, theta, cot, s, grad);
    }

    /// The local pullback against precomputed transformer Hessians. Needs
    /// `prepare`, and `s.t`, `s.l` set from `dense`.
    fn local_pullback_dense(
        &self,
        shape: Shape,
        theta: &[f64],
        dense: &DenseTransformer,
        cot: &Cotangent,
        s: &mut Scratch,
        grad: &mut [f64],
    ) {
        let n_z = 1 + self.n_par;
        self.pullback_directions(shape, theta, cot, s);
        let mut dir_t = [0.0; N];
        let mut dir_l = [0.0; N];
        dir_t[0] = cot.delta;
        dir_t[1..n_z].copy_from_slice(&s.t_bar);
        dir_l[0] = cot.mu;
        dir_l[1..n_z].copy_from_slice(&s.l_bar);
        for m in 0..self.n_par {
            let mut value = cot.x * s.t[m];
            for n in 0..n_z {
                value += dense.hess_t[n][1 + m] * dir_t[n] + dense.hess_l[n][1 + m] * dir_l[n];
            }
            s.pi_bar[m] = value;
        }
        self.pullback_finish(shape, theta, cot, s, grad);
    }

    // ------------------------------------------------------------ global sweeps

    /// Solves `J^T w = b` in place (back substitution, children first).
    fn solve_transpose(&self, edge_a: &[f64], delta: &[f64], b: &mut [f64]) {
        for i in (0..self.n_var).rev() {
            let w_i = b[i] / delta[i];
            b[i] = w_i;
            for e in self.edges(i) {
                b[self.parent_index[e]] -= edge_a[e] * w_i;
            }
        }
    }

    /// Solves `J z = b` in place (forward substitution, parents first).
    fn solve(&self, edge_a: &[f64], delta: &[f64], b: &mut [f64]) {
        for i in 0..self.n_var {
            let mut acc = b[i];
            for e in self.edges(i) {
                acc -= edge_a[e] * b[self.parent_index[e]];
            }
            b[i] = acc / delta[i];
        }
    }

    fn primal_draw(
        &self,
        theta: &[f64],
        draw: usize,
        tape: DrawTapeMut,
        out: &mut [f64],
        s: &mut DrawScratch,
    ) {
        let n_var = self.n_var;
        let y = &self.y[draw * n_var..(draw + 1) * n_var];
        let g = &self.g[draw * n_var..(draw + 1) * n_var];
        let [grad_log_det, ..] = &mut s.by_var;
        grad_log_det.fill(0.0);

        for i in 0..n_var {
            self.gather_parents(i, y, &mut s.local.y_parents);
            let edges = self.edges(i);
            let (x, delta, mu) = self.local_primal(
                i,
                &theta[self.params(i)],
                y[i],
                &mut tape.a[i * self.n_unit..(i + 1) * self.n_unit],
                &mut tape.edge_a[edges.clone()],
                &mut tape.edge_l[edges.clone()],
                &mut s.local,
            );
            tape.x[i] = x;
            tape.delta[i] = delta;
            grad_log_det[i] += mu;
            for e in edges {
                grad_log_det[self.parent_index[e]] += tape.edge_l[e];
            }
        }

        for i in 0..n_var {
            tape.w[i] = g[i] - grad_log_det[i];
        }
        self.solve_transpose(tape.edge_a, tape.delta, tape.w);

        let scale = self.scale();
        for i in 0..n_var {
            out[i] = scale * (tape.x[i] + tape.w[i]);
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.edges(i) {
                    out[n_var + e] = scale * reg * (tape.edge_l[e] - tape.x[i] * tape.edge_a[e]);
                }
            }
        }
    }

    fn pushforward_draw(
        &self,
        theta: &[f64],
        v: &[f64],
        draw: usize,
        tape: &DrawTape,
        out: &mut [f64],
        s: &mut DrawScratch,
    ) {
        let n_var = self.n_var;
        let y = &self.y[draw * n_var..(draw + 1) * n_var];
        let [x_dot, rhs, ..] = &mut s.by_var;
        let [a_dot, l_dot] = &mut s.by_edge;
        rhs.fill(0.0);

        for i in 0..n_var {
            self.gather_parents(i, y, &mut s.local.y_parents);
            let edges = self.edges(i);
            let params = self.params(i);
            let (xd, delta_dot, mu_dot) = self.local_pushforward(
                i,
                &theta[params.clone()],
                &v[params],
                y[i],
                &tape.a[i * self.n_unit..(i + 1) * self.n_unit],
                &mut a_dot[edges.clone()],
                &mut l_dot[edges.clone()],
                &mut s.local,
            );
            x_dot[i] = xd;
            // J^T w_dot = -(grad_y log_det)_dot - J_dot^T w
            rhs[i] -= mu_dot + delta_dot * tape.w[i];
            for e in edges {
                rhs[self.parent_index[e]] -= l_dot[e] + a_dot[e] * tape.w[i];
            }
        }
        self.solve_transpose(tape.edge_a, tape.delta, rhs);

        let scale = self.scale();
        for i in 0..n_var {
            out[i] = scale * (x_dot[i] + rhs[i]);
        }
        if let Some(reg) = self.regularization {
            for i in 0..n_var {
                for e in self.edges(i) {
                    out[n_var + e] =
                        scale * reg * (l_dot[e] - x_dot[i] * tape.edge_a[e] - tape.x[i] * a_dot[e]);
                }
            }
        }
    }

    /// Accumulates this draw's `J^T r_bar` into `grad`.
    fn pullback_draw(
        &self,
        theta: &[f64],
        r_bar: &[f64],
        draw: usize,
        tape: &DrawTape,
        grad: &mut [f64],
        s: &mut DrawScratch,
    ) {
        let n_var = self.n_var;
        let y = &self.y[draw * n_var..(draw + 1) * n_var];
        let scale = self.scale();
        let [z, ..] = &mut s.by_var;
        let [cot_a, cot_l] = &mut s.by_edge;

        for i in 0..n_var {
            z[i] = scale * r_bar[i];
        }
        self.solve(tape.edge_a, tape.delta, z);

        for i in 0..n_var {
            self.gather_parents(i, y, &mut s.local.y_parents);
            let edges = self.edges(i);
            let mut x_bar = scale * r_bar[i];
            for e in edges.clone() {
                let z_p = z[self.parent_index[e]];
                cot_a[e] = -tape.w[i] * z_p;
                cot_l[e] = -z_p;
                if let Some(reg) = self.regularization {
                    let o = scale * reg * r_bar[n_var + e];
                    x_bar -= o * tape.edge_a[e];
                    cot_a[e] -= tape.x[i] * o;
                    cot_l[e] += o;
                }
            }
            let cot = Cotangent {
                x: x_bar,
                delta: -tape.w[i] * z[i],
                mu: -z[i],
                edge_a: &cot_a[edges.clone()],
                edge_l: &cot_l[edges],
            };
            let params = self.params(i);
            self.local_pullback(
                i,
                &theta[params.clone()],
                y[i],
                &tape.a[i * self.n_unit..(i + 1) * self.n_unit],
                &cot,
                &mut s.local,
                &mut grad[params],
            );
        }
    }

    // ------------------------------------------------------------ operators

    fn check_params(&self, values: &[f64], name: &str) -> Result<()> {
        if values.len() != self.n_params() {
            bail!(
                "{name} has length {}, expected {}",
                values.len(),
                self.n_params()
            );
        }
        Ok(())
    }

    /// `(n_draw, n_residuals)` residuals at `theta`, recording the tape every
    /// derivative reads if `record`.
    fn residuals(&mut self, theta: &[f64], record: bool) -> Result<Vec<f64>> {
        self.check_params(theta, "theta")?;
        if self.n_draw == 0 {
            bail!("no data: call `set_data` first");
        }
        let (n_draw, n_var, n_edge, n_unit) = (self.n_draw, self.n_var, self.n_edge(), self.n_unit);
        let mut tape = Tape {
            theta: theta.to_vec(),
            a: vec![0.0; n_draw * n_var * n_unit],
            edge_a: vec![0.0; n_draw * n_edge],
            edge_l: vec![0.0; n_draw * n_edge],
            delta: vec![0.0; n_draw * n_var],
            x: vec![0.0; n_draw * n_var],
            w: vec![0.0; n_draw * n_var],
        };
        let n_res = self.n_residuals();
        let mut out = vec![0.0; n_draw * n_res];
        {
            let views: Vec<_> = chunks(&mut tape.a, n_var * n_unit, n_draw)
                .into_iter()
                .zip(chunks(&mut tape.edge_a, n_edge, n_draw))
                .zip(chunks(&mut tape.edge_l, n_edge, n_draw))
                .zip(chunks(&mut tape.delta, n_var, n_draw))
                .zip(chunks(&mut tape.x, n_var, n_draw))
                .zip(chunks(&mut tape.w, n_var, n_draw))
                .zip(chunks(&mut out, n_res, n_draw))
                .enumerate()
                .map(|(draw, ((((((a, edge_a), edge_l), delta), x), w), out))| {
                    (
                        draw,
                        DrawTapeMut {
                            a,
                            edge_a,
                            edge_l,
                            delta,
                            x,
                            w,
                        },
                        out,
                    )
                })
                .collect();
            views.into_par_iter().for_each_init(
                || DrawScratch::new(self),
                |s, (draw, tape, out)| self.primal_draw(theta, draw, tape, out, s),
            );
        }
        if record {
            self.tape = Some(tape);
        }
        Ok(out)
    }

    fn pushforward(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        let tape = self.tape()?;
        let n_res = self.n_residuals();
        let mut out = vec![0.0; self.n_draw * n_res];
        out.par_chunks_mut(n_res.max(1)).enumerate().for_each_init(
            || DrawScratch::new(self),
            |s, (draw, out)| {
                let draw_tape = self.draw_tape(tape, draw);
                self.pushforward_draw(&tape.theta, v, draw, &draw_tape, out, s)
            },
        );
        Ok(out)
    }

    fn pullback(&self, r_bar: &[f64]) -> Result<Vec<f64>> {
        let n_res = self.n_residuals();
        if r_bar.len() != self.n_draw * n_res {
            bail!("r_bar must have shape ({}, {n_res})", self.n_draw);
        }
        let tape = self.tape()?;
        Ok(self.sum_over_draws(|draw, grad, s| {
            let draw_tape = self.draw_tape(tape, draw);
            let r_bar = &r_bar[draw * n_res..(draw + 1) * n_res];
            self.pullback_draw(&tape.theta, r_bar, draw, &draw_tape, grad, s);
        }))
    }

    /// `J^T J v`, one draw at a time, without forming `J v` for all draws.
    fn gauss_newton_product(&self, v: &[f64]) -> Result<Vec<f64>> {
        self.check_params(v, "v")?;
        let tape = self.tape()?;
        Ok(self.sum_over_draws(|draw, grad, s| {
            let draw_tape = self.draw_tape(tape, draw);
            let mut residual = std::mem::take(&mut s.residual);
            self.pushforward_draw(&tape.theta, v, draw, &draw_tape, &mut residual, s);
            self.pullback_draw(&tape.theta, &residual, draw, &draw_tape, grad, s);
            s.residual = residual;
        }))
    }

    fn sum_over_draws(&self, f: impl Fn(usize, &mut [f64], &mut DrawScratch) + Sync) -> Vec<f64> {
        let n_params = self.n_params();
        (0..self.n_draw)
            .into_par_iter()
            .fold(
                || (vec![0.0; n_params], DrawScratch::new(self)),
                |(mut grad, mut s), draw| {
                    f(draw, &mut grad, &mut s);
                    (grad, s)
                },
            )
            .map(|(grad, _)| grad)
            .reduce(
                || vec![0.0; n_params],
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(&b) {
                        *a += b;
                    }
                    a
                },
            )
    }

    // ------------------------------------------------------------ exact blocks

    /// `Sigma = (J^T J)^{-1}` on the diagonal and the filled pattern, by
    /// Takahashi's recurrence: the diagonal, then one slot per entry of
    /// `filled_index`.
    fn selected_inverse(&self, edge_a: &[f64], delta: &[f64], store: &mut [f64]) {
        for i in 0..self.n_var {
            let filled = self.filled_indptr[i]..self.filled_indptr[i + 1];
            for slot in filled {
                let j = self.filled_index[slot];
                let mut acc = 0.0;
                for e in self.edges(i) {
                    acc += edge_a[e] * self.sigma(store, self.parent_index[e], j);
                }
                store[self.n_var + slot] = -acc / delta[i];
            }
            let mut acc = 0.0;
            for e in self.edges(i) {
                acc += edge_a[e] * self.sigma(store, i, self.parent_index[e]);
            }
            store[i] = (1.0 / delta[i] - acc) / delta[i];
        }
    }

    fn sigma(&self, store: &[f64], a: usize, b: usize) -> f64 {
        if a == b {
            return store[a];
        }
        let (row, col) = if a > b { (a, b) } else { (b, a) };
        let start = self.filled_indptr[row];
        let filled = &self.filled_index[start..self.filled_indptr[row + 1]];
        let k = filled
            .binary_search(&col)
            .expect("selected inverse entry outside the filled pattern");
        store[self.n_var + start + k]
    }

    /// Sub-blocks of variable `i`'s slice: consecutive runs of at most
    /// `max_block_size` parameters.
    fn block_ranges(&self, i: usize, max_block_size: usize) -> Vec<std::ops::Range<usize>> {
        let size = self.shape(i).size();
        (0..size)
            .step_by(max_block_size)
            .map(|start| start..(start + max_block_size).min(size))
            .collect()
    }

    /// Exact Gauss-Newton blocks, `sum_draws V_i^T V_i / n_draw` restricted
    /// to the sub-blocks of `block_ranges`. Returns, per sub-block, its first
    /// global parameter index and size, and the blocks as concatenated
    /// row-major squares.
    fn gauss_newton_blocks(
        &self,
        max_block_size: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<f64>)> {
        if max_block_size == 0 {
            bail!("max_block_size must be positive");
        }
        let tape = self.tape()?;
        let n_store = self.n_var + self.filled_index.len();
        let mut stores = vec![0.0; self.n_draw * n_store];
        stores
            .par_chunks_mut(n_store)
            .enumerate()
            .for_each(|(draw, store)| {
                let draw_tape = self.draw_tape(tape, draw);
                self.selected_inverse(draw_tape.edge_a, draw_tape.delta, store);
            });

        let per_var: Vec<(Vec<std::ops::Range<usize>>, Vec<f64>)> = (0..self.n_var)
            .into_par_iter()
            .map_init(
                || self.scratch(),
                |s, i| self.variable_blocks(tape, &stores, n_store, i, max_block_size, s),
            )
            .collect();

        let mut starts = Vec::new();
        let mut sizes = Vec::new();
        let mut data = Vec::new();
        for (i, (ranges, blocks)) in per_var.into_iter().enumerate() {
            for range in ranges {
                starts.push(self.param_offset[i] + range.start);
                sizes.push(range.len());
            }
            data.extend(blocks);
        }
        Ok((starts, sizes, data))
    }

    fn variable_blocks(
        &self,
        tape: &Tape,
        stores: &[f64],
        n_store: usize,
        i: usize,
        max_block_size: usize,
        s: &mut Scratch,
    ) -> (Vec<std::ops::Range<usize>>, Vec<f64>) {
        let shape = self.shape(i);
        let n_theta = shape.size();
        let n_p = shape.n_parent;
        let n_s = n_p + 1;
        let theta = &tape.theta[self.params(i)];
        let parents = &self.parent_index[self.edges(i)];
        let n_score = if self.regularization.is_some() {
            n_p
        } else {
            0
        };
        let n_rows = 1 + n_s + n_score;

        let ranges = self.block_ranges(i, max_block_size);
        let block_offset: Vec<usize> = ranges
            .iter()
            .scan(0, |acc, r| {
                let start = *acc;
                *acc += r.len() * r.len();
                Some(start)
            })
            .collect();
        let mut blocks = vec![0.0; ranges.iter().map(|r| r.len() * r.len()).sum()];

        let mut rows = vec![0.0; n_rows * n_theta];
        let mut v = vec![0.0; n_rows * n_theta];
        let mut cot_a = vec![0.0; n_p];
        let mut cot_l = vec![0.0; n_p];
        let mut k_mat = vec![0.0; n_s * n_s];
        let mut d = vec![0.0; n_s];
        let mut index = Vec::with_capacity(n_s);
        index.push(i);
        index.extend_from_slice(parents);

        for draw in 0..self.n_draw {
            let draw_tape = self.draw_tape(tape, draw);
            let y = &self.y[draw * self.n_var..(draw + 1) * self.n_var];
            self.gather_parents(i, y, &mut s.y_parents);
            let a = &draw_tape.a[i * self.n_unit..(i + 1) * self.n_unit];
            self.prepare(shape, theta, a, s);
            let dense = self.dense_transformer(y[i], &s.pi);
            s.t.copy_from_slice(&dense.t[1..1 + self.n_par]);
            s.l.copy_from_slice(&dense.l[1..1 + self.n_par]);

            let w_i = draw_tape.w[i];
            let x_i = draw_tape.x[i];
            let edge_a = &draw_tape.edge_a[self.edges(i)];

            // Each row of `[a; B; scores]` is a local pullback of one seed.
            rows.fill(0.0);
            let mut seed = |row: usize,
                            x: f64,
                            delta: f64,
                            mu: f64,
                            parent: Option<(usize, f64, f64)>,
                            s: &mut Scratch| {
                cot_a.fill(0.0);
                cot_l.fill(0.0);
                if let Some((j, value_a, value_l)) = parent {
                    cot_a[j] = value_a;
                    cot_l[j] = value_l;
                }
                let cot = Cotangent {
                    x,
                    delta,
                    mu,
                    edge_a: &cot_a,
                    edge_l: &cot_l,
                };
                let out = &mut rows[row * n_theta..(row + 1) * n_theta];
                self.local_pullback_dense(shape, theta, &dense, &cot, s, out);
            };
            // a = dx_i / dtheta_i
            seed(0, 1.0, 0.0, 0.0, None, s);
            // q_own = -mu_i - delta_i w_i
            seed(1, 0.0, -w_i, -1.0, None, s);
            // q_j = -L[i,j] - A[i,j] w_i
            for j in 0..n_p {
                seed(2 + j, 0.0, 0.0, 0.0, Some((j, -w_i, -1.0)), s);
            }
            // score_j = L[i,j] - x_i A[i,j]
            for j in 0..n_score {
                seed(1 + n_s + j, -edge_a[j], 0.0, 0.0, Some((j, -x_i, 1.0)), s);
            }

            // K = Sigma on {i} + P(i), and its Cholesky factor.
            let store = &stores[draw * n_store..(draw + 1) * n_store];
            for r in 0..n_s {
                for c in 0..n_s {
                    k_mat[r * n_s + c] = self.sigma(store, index[r], index[c]);
                }
            }
            let mut factor = cholesky(&k_mat, n_s);
            if factor.is_none() {
                if let Some(jitter) = self.cholesky_jitter {
                    let max_diag = (0..n_s).map(|r| k_mat[r * n_s + r]).fold(0.0, f64::max);
                    let mut jittered = k_mat.clone();
                    for r in 0..n_s {
                        jittered[r * n_s + r] += jitter * max_diag;
                    }
                    factor = cholesky(&jittered, n_s);
                }
            }
            let factor = factor.unwrap_or_else(|| vec![f64::NAN; n_s * n_s]);

            // d = L^{-1} e_0 / delta_i
            for r in 0..n_s {
                let mut acc = if r == 0 {
                    1.0 / draw_tape.delta[i]
                } else {
                    0.0
                };
                for c in 0..r {
                    acc -= factor[r * n_s + c] * d[c];
                }
                d[r] = acc / factor[r * n_s + r];
            }
            let outside = (1.0 - dot(&d, &d)).max(0.0).sqrt();

            // V = [outside a; L^T B + d a; sqrt(rho) scores]
            let (a_row, rest) = rows.split_at(n_theta);
            v.fill(0.0);
            for (out, &value) in v[..n_theta].iter_mut().zip(a_row) {
                *out = outside * value;
            }
            for r in 0..n_s {
                let out = &mut v[(1 + r) * n_theta..(2 + r) * n_theta];
                axpy(d[r], a_row, out);
                for m in r..n_s {
                    axpy(
                        factor[m * n_s + r],
                        &rest[m * n_theta..(m + 1) * n_theta],
                        out,
                    );
                }
            }
            if let Some(reg) = self.regularization {
                for j in 0..n_score {
                    let row = 1 + n_s + j;
                    let out = &mut v[row * n_theta..(row + 1) * n_theta];
                    axpy(reg, &rows[row * n_theta..(row + 1) * n_theta], out);
                }
            }

            for (range, &offset) in ranges.iter().zip(&block_offset) {
                let size = range.len();
                let block = &mut blocks[offset..offset + size * size];
                for row in v.chunks_exact(n_theta) {
                    let row = &row[range.clone()];
                    for (r, &value) in row.iter().enumerate() {
                        if value != 0.0 {
                            axpy(value, &row[..=r], &mut block[r * size..r * size + r + 1]);
                        }
                    }
                }
            }
        }

        let inv_n = 1.0 / self.n_draw as f64;
        for (range, &offset) in ranges.iter().zip(&block_offset) {
            let size = range.len();
            let block = &mut blocks[offset..offset + size * size];
            for r in 0..size {
                for c in 0..=r {
                    let value = block[r * size + c] * inv_n;
                    block[r * size + c] = value;
                    block[c * size + r] = value;
                }
            }
        }
        (ranges, blocks)
    }
}

/// Lower Cholesky factor of a small row-major SPD matrix, or `None`.
fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut factor = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..=r {
            let mut acc = matrix[r * n + c];
            for k in 0..c {
                acc -= factor[r * n + k] * factor[c * n + k];
            }
            if r == c {
                if acc.is_nan() || acc <= 0.0 {
                    return None;
                }
                factor[r * n + r] = acc.sqrt();
            } else {
                factor[r * n + c] = acc / factor[c * n + c];
            }
        }
    }
    Some(factor)
}

fn as_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| anyhow::anyhow!("negative index {v}")))
        .collect()
}

/// Residuals of the LM fit of a `SparseTriangularMap` and their derivatives,
/// see `nutpie.triangular_lm`.
#[pyclass(name = "FisherResiduals")]
pub struct PyFisherResiduals {
    inner: FisherResiduals,
}

#[pymethods]
impl PyFisherResiduals {
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        n_unit,
        n_par,
        location_index,
        transformer,
        fisher_regularization = None,
        cholesky_jitter = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        n_unit: usize,
        n_par: usize,
        location_index: usize,
        transformer: &Bound<'_, PyAny>,
        fisher_regularization: Option<f64>,
        cholesky_jitter: Option<f64>,
    ) -> Result<Self> {
        let specs: Vec<Contract2Spec> = pythonize::depythonize(transformer)?;
        Ok(Self {
            inner: FisherResiduals::new(
                as_usize(parent_indptr.as_slice()?)?,
                as_usize(parent_index.as_slice()?)?,
                n_unit,
                n_par,
                location_index,
                specs,
                fisher_regularization,
                cholesky_jitter,
            )?,
        })
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params()
    }

    #[getter]
    fn n_residuals(&self) -> usize {
        self.inner.n_residuals()
    }

    #[getter]
    fn n_draw(&self) -> usize {
        self.inner.n_draw
    }

    /// Start of each variable's parameter slice, plus the total.
    #[getter]
    fn param_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(
            py,
            self.inner.param_offset.iter().map(|&o| o as i64).collect(),
        )
    }

    /// Set the map-space draws and gradients, each ``(n_draw, n_var)``
    /// flattened. Drops the tape.
    fn set_data(
        &mut self,
        y: PyReadonlyArray1<'_, f64>,
        g: PyReadonlyArray1<'_, f64>,
    ) -> Result<()> {
        self.inner
            .set_data(y.as_slice()?.to_vec(), g.as_slice()?.to_vec())
    }

    /// Flattened ``(n_draw, n_residuals)`` residuals at `theta`; with
    /// `record`, keeps the tape the derivatives below use.
    #[pyo3(signature = (theta, record = true))]
    fn residuals<'py>(
        &mut self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
        record: bool,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let theta = theta.as_slice()?.to_vec();
        let out = py.detach(|| self.inner.residuals(&theta, record))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J v``, flattened ``(n_draw, n_residuals)``, at the recorded `theta`.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let out = py.detach(|| self.inner.pushforward(&v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T r_bar`` for a flattened ``(n_draw, n_residuals)`` `r_bar`.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        r_bar: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let r_bar = r_bar.as_slice()?.to_vec();
        let out = py.detach(|| self.inner.pullback(&r_bar))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// ``J^T J v``.
    fn gauss_newton_product<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let v = v.as_slice()?.to_vec();
        let out = py.detach(|| self.inner.gauss_newton_product(&v))?;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Exact Gauss-Newton sub-blocks: ``(starts, sizes, data)``, where block
    /// ``k`` covers parameters ``starts[k]:starts[k] + sizes[k]`` and is
    /// stored row-major in `data`, after the blocks before it.
    fn gauss_newton_blocks<'py>(
        &self,
        py: Python<'py>,
        max_block_size: usize,
    ) -> Result<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let (starts, sizes, data) = py.detach(|| self.inner.gauss_newton_blocks(max_block_size))?;
        Ok((
            PyArray1::from_vec(py, starts.into_iter().map(|v| v as i64).collect()),
            PyArray1::from_vec(py, sizes.into_iter().map(|v| v as i64).collect()),
            PyArray1::from_vec(py, data),
        ))
    }
}
