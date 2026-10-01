//! Levenberg-Marquardt on [`FisherResiduals`], a port of `nutpie.lmopt.step`
//! restricted to what the triangular flow fit uses: exact Gauss-Newton
//! blocks, Marquardt damping, the parabolic line search, and the conditioners
//! only (the affine is frozen).
//!
//! Each step:
//!
//! 1. Rebuild the exact blocks if due, replace non-finite sub-blocks by their
//!    diagonal, and set the Marquardt floors and `D = max(diag H, floor)`.
//! 2. Factor the damped sub-blocks `H + lam diag(D)` as the preconditioner,
//!    exactly the block diagonal of the CG operator. A sub-block that is
//!    indefinite from rounding falls back to its damped diagonal.
//! 3. Solve `(J^T J + lam D) p = -J^T r` by preconditioned CG, optionally
//!    with a limited-memory preconditioner from earlier solves on top.
//! 4. Evaluate `theta + p`, shorten the step to a fitted parabola's minimum
//!    if that is better, and accept or reject on the gain ratio.
//! 5. Update `lam` (Nielsen) and the CG forcing term (Eisenstat-Walker),
//!    unless CG stops on the quadratic model instead.

use std::collections::VecDeque;

use anyhow::{bail, Result};
use faer::linalg::solvers::Solve;
use faer::{Mat, MatMut, Side};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use serde::Serialize;

use crate::triangular_lm::{FisherResiduals, GnBlock, PyFisherResiduals, Tape};

/// Marquardt floor, as a fraction of each conditioner's largest curvature.
const MARQUARDT_FLOOR: f64 = 1e-4;
/// Fraction of the global largest curvature, for conditioners flat as a whole.
const MARQUARDT_GLOBAL_FLOOR: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Forcing {
    Residual,
    Rho,
    /// Stop CG once it barely lowers the quadratic model; the residual
    /// test only caps it at `cg_tol`.
    Model,
}

#[derive(Debug, Clone)]
struct Settings {
    cg_max: usize,
    cg_tol: f64,
    /// With `Forcing::Model`, the Nash-Sofer threshold.
    cg_model_tol: f64,
    /// With `Forcing::Model`, the fixed cost of an LM step besides CG, in
    /// Gauss-Newton products; 0 is plain Nash-Sofer.
    cg_model_cost: f64,
    cg_eta_max: f64,
    cg_gamma: f64,
    cg_alpha: f64,
    accept_rho: f64,
    nu0: f64,
    lam_min: f64,
    lam_max: f64,
    lam_lo_decay: f64,
    /// `lam_lo` is reset after two good, unshortened full steps in a row
    /// with `|rho_full - 1|` below this; 0 never resets.
    lam_lo_reset: f64,
    ls_min_fraction: f64,
    forcing: Forcing,
    max_block_size: usize,
    rebuild_every: usize,
    /// CG directions kept for the limited-memory preconditioner; 0 disables
    /// it.
    lmp_size: usize,
    /// Directions of `S^T A S` with eigenvalues below `lmp_tol` times the
    /// largest are dropped.
    lmp_tol: f64,
}

/// Exact Gauss-Newton sub-blocks, sanitized, with their Marquardt diagonal.
struct Blocks {
    blocks: Vec<GnBlock>,
    /// Marquardt floor of the variable each sub-block belongs to.
    floor: Vec<f64>,
    /// `max(diag H, floor)`, per parameter.
    diagonal: Vec<f64>,
    nonfinite: usize,
}

/// One factored, damped sub-block, or its damped diagonal if the Cholesky
/// failed.
enum Inverse {
    Cholesky(faer::linalg::solvers::Llt<f64>),
    Diagonal(Vec<f64>),
}

/// A CG search direction `s` and its undamped product `H s = J^T J s`.
/// Stored without the damping, so `A s = H s + lam D s` stays exact when
/// `lam` or `D` change; only a change of `theta` makes `H s` stale.
struct Pair {
    s: Vec<f64>,
    hs: Vec<f64>,
}

/// Limited-memory preconditioner (Gratton, Sartenaer and Tshimanga 2011)
/// around the block preconditioner `M`:
///
/// ```text
/// P = (I - W (AW)^T) M^-1 (I - AW W^T) + W W^T,
/// ```
///
/// with `W` an `A`-orthonormal basis of the stored directions. `P A` is the
/// identity on `span W`, which deflates the modes earlier solves already
/// found, and `P` is SPD for any SPD `M`.
struct Lmp {
    w: Vec<Vec<f64>>,
    aw: Vec<Vec<f64>>,
}

impl Lmp {
    /// From the stored pairs at the current `lam` and `D`: orthonormalize
    /// in `S^T A S` by an eigendecomposition, dropping near-dependent
    /// directions. `None` if nothing is left.
    fn new(pairs: &VecDeque<Pair>, diagonal: &[f64], lam: f64, tol: f64) -> Option<Self> {
        let k = pairs.len();
        if k == 0 {
            return None;
        }
        let a_s: Vec<Vec<f64>> = pairs
            .iter()
            .map(|pair| {
                pair.hs
                    .iter()
                    .zip(&pair.s)
                    .zip(diagonal)
                    .map(|((h, s), d)| h + lam * d * s)
                    .collect()
            })
            .collect();
        let mut g = Mat::<f64>::zeros(k, k);
        for a in 0..k {
            for b in 0..=a {
                let v = 0.5 * (dot(&pairs[a].s, &a_s[b]) + dot(&pairs[b].s, &a_s[a]));
                g[(a, b)] = v;
                g[(b, a)] = v;
            }
        }
        if !(0..k).all(|c| g.col_as_slice(c).iter().all(|v| v.is_finite())) {
            return None;
        }
        let eigen = g.self_adjoint_eigen(Side::Lower).ok()?;
        let values = eigen.S().column_vector();
        let vectors = eigen.U();
        // Ascending order: the largest is last.
        let largest = values[k - 1];
        if largest <= 0.0 {
            return None;
        }
        let n = diagonal.len();
        let (mut w, mut aw) = (Vec::new(), Vec::new());
        for l in 0..k {
            if values[l] <= tol * largest {
                continue;
            }
            let scale = 1.0 / values[l].sqrt();
            let (mut wl, mut awl) = (vec![0.0; n], vec![0.0; n]);
            for j in 0..k {
                let c = vectors[(j, l)] * scale;
                axpy(c, &pairs[j].s, &mut wl);
                axpy(c, &a_s[j], &mut awl);
            }
            w.push(wl);
            aw.push(awl);
        }
        if w.is_empty() {
            None
        } else {
            Some(Self { w, aw })
        }
    }

    fn rank(&self) -> usize {
        self.w.len()
    }

    fn apply(&self, v: &[f64], minv: impl Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
        let c: Vec<f64> = self.w.iter().map(|w| dot(w, v)).collect();
        let mut u = v.to_vec();
        for (c, aw) in c.iter().zip(&self.aw) {
            axpy(-c, aw, &mut u);
        }
        let mut z = minv(&u);
        let d: Vec<f64> = self.aw.iter().map(|aw| dot(aw, &z)).collect();
        for ((c, d), w) in c.iter().zip(&d).zip(&self.w) {
            axpy(c - d, w, &mut z);
        }
        z
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepInfo {
    #[serde(rename = "F")]
    f: f64,
    #[serde(rename = "F_new")]
    f_new: f64,
    #[serde(rename = "F_out")]
    f_out: f64,
    accept: bool,
    rho: f64,
    rho_full: f64,
    actual: f64,
    pred: f64,
    lam_in: f64,
    lam_out: f64,
    lam_nu: f64,
    n_cg: usize,
    cg_eta: f64,
    cg_converged: bool,
    grad_norm: f64,
    step_norm: f64,
    full_step_norm: f64,
    finite: bool,
    rebuilt_blocks: bool,
    step_length: f64,
    nonfinite_blocks: usize,
    failed_inverses: usize,
    lmp_rank: usize,
}

pub struct LmOptimizer {
    problem: FisherResiduals,
    settings: Settings,
    theta: Vec<f64>,
    tape: Tape,
    r: Vec<f64>,
    lam: f64,
    nu: f64,
    lam_lo: f64,
    /// Consecutive clean full steps, see `step`.
    clean_steps: usize,
    eta: f64,
    p_prev: Vec<f64>,
    blocks: Option<Blocks>,
    accepted_since_build: usize,
    /// The latest `lmp_size` CG directions, oldest first.
    lmp_pairs: VecDeque<Pair>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// `x / y`, with `y` replaced by `1e-30` where it is tiny, as `lmopt`.
fn safe_ratio(x: f64, y: f64) -> f64 {
    x / if y.abs() < 1e-30 { 1e-30 } else { y }
}

impl LmOptimizer {
    fn new(
        problem: FisherResiduals,
        theta: Vec<f64>,
        lam: f64,
        settings: Settings,
    ) -> Result<Self> {
        if theta.len() != problem.n_params() {
            bail!(
                "theta has length {}, expected {}",
                theta.len(),
                problem.n_params()
            );
        }
        if problem.n_draw == 0 {
            bail!("the residuals have no data");
        }
        if settings.max_block_size == 0 || settings.rebuild_every == 0 {
            bail!("max_block_size and rebuild_every must be positive");
        }
        let (r, tape) = problem.residuals(&theta)?;
        let n_params = theta.len();
        Ok(Self {
            problem,
            theta,
            tape,
            r,
            lam,
            nu: settings.nu0,
            lam_lo: 0.0,
            clean_steps: 0,
            eta: settings.cg_eta_max,
            p_prev: vec![0.0; n_params],
            blocks: None,
            accepted_since_build: settings.rebuild_every,
            lmp_pairs: VecDeque::new(),
            settings,
        })
    }

    fn build_blocks(&self) -> Result<Blocks> {
        let mut blocks = self
            .problem
            .gauss_newton_blocks(&self.tape, self.settings.max_block_size)?;

        // Non-finite sub-blocks fall back to their finite diagonal; one would
        // otherwise spread through the global floor into every damping.
        let mut nonfinite = 0;
        for block in &mut blocks {
            let m = &mut block.matrix;
            let size = m.nrows();
            let finite = (0..size).all(|c| m.col_as_slice(c).iter().all(|v| v.is_finite()));
            if finite {
                continue;
            }
            nonfinite += 1;
            let diagonal: Vec<f64> = (0..size)
                .map(|k| {
                    if m[(k, k)].is_finite() {
                        m[(k, k)]
                    } else {
                        0.0
                    }
                })
                .collect();
            *m = Mat::from_fn(size, size, |r, c| if r == c { diagonal[r] } else { 0.0 });
        }

        // Marquardt floors per variable (conditioner).
        let param_offset = &self.problem.param_offset;
        let var_of = |start: usize| param_offset.partition_point(|&o| o <= start) - 1;
        let mut local = vec![f64::NEG_INFINITY; self.problem.n_var];
        for block in &blocks {
            let var = var_of(block.start);
            for k in 0..block.matrix.nrows() {
                local[var] = local[var].max(block.matrix[(k, k)]);
            }
        }
        let global = local
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-300);
        let floor_of_var: Vec<f64> = local
            .iter()
            .map(|&l| (MARQUARDT_FLOOR * l).max(MARQUARDT_GLOBAL_FLOOR * global))
            .collect();

        let mut floor = Vec::with_capacity(blocks.len());
        let mut diagonal = vec![0.0; self.theta.len()];
        for block in &blocks {
            let var_floor = floor_of_var[var_of(block.start)];
            floor.push(var_floor);
            for k in 0..block.matrix.nrows() {
                diagonal[block.start + k] = block.matrix[(k, k)].max(var_floor);
            }
        }
        Ok(Blocks {
            blocks,
            floor,
            diagonal,
            nonfinite,
        })
    }

    /// Factors of the damped sub-blocks, and how many fell back to the
    /// diagonal.
    fn preconditioner(&self, blocks: &Blocks, lam: f64) -> (Vec<Inverse>, usize) {
        use rayon::prelude::*;
        let inverses: Vec<Inverse> = blocks
            .blocks
            .par_iter()
            .zip(&blocks.floor)
            .map(|(block, &floor)| {
                let m = &block.matrix;
                let size = m.nrows();
                let mut damped = m.clone();
                for k in 0..size {
                    damped[(k, k)] += lam * m[(k, k)].max(floor);
                }
                match damped.llt(Side::Lower) {
                    Ok(llt)
                        if llt
                            .L()
                            .col_iter()
                            .all(|col| col.iter().all(|v| v.is_finite())) =>
                    {
                        Inverse::Cholesky(llt)
                    }
                    _ => Inverse::Diagonal((0..size).map(|k| 1.0 / damped[(k, k)]).collect()),
                }
            })
            .collect();
        let failed = inverses
            .iter()
            .filter(|inv| matches!(inv, Inverse::Diagonal(_)))
            .count();
        (inverses, failed)
    }

    fn apply_preconditioner(blocks: &Blocks, inverses: &[Inverse], v: &[f64]) -> Vec<f64> {
        let mut out = v.to_vec();
        for (block, inverse) in blocks.blocks.iter().zip(inverses) {
            let size = block.matrix.nrows();
            let slice = &mut out[block.start..block.start + size];
            match inverse {
                Inverse::Cholesky(llt) => {
                    llt.solve_in_place(MatMut::from_column_major_slice_mut(slice, size, 1));
                }
                Inverse::Diagonal(inv) => {
                    for (v, d) in slice.iter_mut().zip(inv) {
                        *v *= d;
                    }
                }
            }
        }
        out
    }

    /// `(J^T J + lam D) v`.
    fn damped_product(&self, diagonal: &[f64], lam: f64, v: &[f64]) -> Result<Vec<f64>> {
        let mut out = self.problem.gauss_newton_product(&self.tape, v)?;
        for ((o, d), v) in out.iter_mut().zip(diagonal).zip(v) {
            *o += lam * d * v;
        }
        Ok(out)
    }

    /// PCG as `lmopt.pcg`: warm start from the best multiple of `x0`, stop
    /// once `||r||_{M^-1} < rtol ||b||_{M^-1}`, with `M^-1` the block
    /// preconditioner, wrapped in `lmp` if given. With `Forcing::Model`,
    /// also stop once iteration `i` lowered the quadratic model `q(x) =
    /// -b.x + x.Ax/2` by less than `cg_model_tol Q / (i + cg_model_cost)`,
    /// with `Q` the decrease of all CG iterations so far: Nash and Sofer
    /// (1990), with the fixed cost of an LM step added to the iteration
    /// count. Returns `(x, n_iters)`,
    /// and the last `lmp_size` search directions with their undamped
    /// products.
    #[allow(clippy::too_many_arguments)]
    fn pcg(
        &self,
        blocks: &Blocks,
        inverses: &[Inverse],
        lmp: Option<&Lmp>,
        lam: f64,
        b: &[f64],
        x0: &[f64],
        rtol: f64,
    ) -> Result<(Vec<f64>, usize, VecDeque<Pair>)> {
        let base = |v: &[f64]| Self::apply_preconditioner(blocks, inverses, v);
        let minv = |v: &[f64]| match lmp {
            Some(lmp) => lmp.apply(v, base),
            None => base(v),
        };
        let av = |v: &[f64]| self.damped_product(&blocks.diagonal, lam, v);
        let lmp_size = self.settings.lmp_size;
        let mut pairs = VecDeque::with_capacity(lmp_size);
        let tol_sq = rtol * rtol * dot(b, &minv(b));
        let model_tol = match self.settings.forcing {
            Forcing::Model => Some(self.settings.cg_model_tol),
            _ => None,
        };
        let model_cost = self.settings.cg_model_cost;

        let mut x = vec![0.0; b.len()];
        let mut r = b.to_vec();
        // How much the CG iterations lowered the quadratic model; each one
        // by `alpha (r.z) / 2`, since `p.r = r.z` in PCG. The warm start is
        // left out: with it, a good `x0` would stop CG after one iteration.
        let mut q_decrease = 0.0;
        if x0.iter().any(|&v| v != 0.0) {
            let ax0 = av(x0)?;
            let (bx, xax) = (dot(b, x0), dot(x0, &ax0));
            let alpha = if bx > 0.0 && xax > 0.0 { bx / xax } else { 0.0 };
            axpy(alpha, x0, &mut x);
            axpy(-alpha, &ax0, &mut r);
        }
        let mut z = minv(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut k = 0;
        while rz > tol_sq && k < self.settings.cg_max && rz.is_finite() {
            let ap = av(&p)?;
            let pap = dot(&p, &ap);
            if lmp_size > 0 && pap > 0.0 && pap.is_finite() {
                if pairs.len() == lmp_size {
                    pairs.pop_front();
                }
                let hs = ap
                    .iter()
                    .zip(&p)
                    .zip(&blocks.diagonal)
                    .map(|((a, p), d)| a - lam * d * p)
                    .collect();
                pairs.push_back(Pair { s: p.clone(), hs });
            }
            let alpha = rz / if pap > 0.0 { pap } else { 1.0 };
            axpy(alpha, &p, &mut x);
            axpy(-alpha, &ap, &mut r);
            k += 1;
            if let Some(model_tol) = model_tol {
                let decrease = 0.5 * alpha * rz;
                q_decrease += decrease;
                if pap > 0.0 && (k as f64 + model_cost) * decrease < model_tol * q_decrease {
                    break;
                }
            }
            z = minv(&r);
            let rz_next = dot(&r, &z);
            let beta = rz_next / rz;
            for (p, z) in p.iter_mut().zip(&z) {
                *p = beta * *p + z;
            }
            rz = rz_next;
        }
        Ok((x, k, pairs))
    }

    fn step(&mut self) -> Result<StepInfo> {
        let s = self.settings.clone();
        let lam = self.lam;

        let rebuild = self.accepted_since_build >= s.rebuild_every;
        if rebuild || self.blocks.is_none() {
            self.blocks = Some(self.build_blocks()?);
            self.accepted_since_build = 0;
        }
        let blocks = self.blocks.as_ref().expect("built above");
        let (inverses, failed_inverses) = self.preconditioner(blocks, lam);
        // Pairs from steps before the last accepted one are stale (`H` was
        // taken at an earlier `theta`), which only slows CG down.
        let lmp = Lmp::new(&self.lmp_pairs, &blocks.diagonal, lam, s.lmp_tol);
        let lmp_rank = lmp.as_ref().map_or(0, Lmp::rank);

        let g = self.problem.pullback(&self.tape, &self.r)?;
        let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
        let eta = match s.forcing {
            Forcing::Model => s.cg_tol,
            _ => self.eta.clamp(s.cg_tol, s.cg_eta_max),
        };
        let (mut p, n_cg, new_pairs) = self.pcg(
            blocks,
            &inverses,
            lmp.as_ref(),
            lam,
            &rhs,
            &self.p_prev,
            eta,
        )?;
        for pair in new_pairs {
            if self.lmp_pairs.len() == s.lmp_size {
                self.lmp_pairs.pop_front();
            }
            self.lmp_pairs.push_back(pair);
        }

        let step_to =
            |p: &[f64]| -> Vec<f64> { self.theta.iter().zip(p).map(|(t, p)| t + p).collect() };
        let (mut r_new, mut tape_new) = self.problem.residuals(&step_to(&p))?;
        let mut jp = self.problem.pushforward(&self.tape, &p)?;

        let f = dot(&self.r, &self.r);
        let mut f_new = dot(&r_new, &r_new);
        let pred_full = -dot(&p, &g) - 0.5 * dot(&jp, &jp);
        let rho_full = safe_ratio(0.5 * (f - f_new), pred_full);
        let full_step_good =
            rho_full > s.accept_rho && pred_full > 0.0 && f_new.is_finite() && rho_full.is_finite();
        let full_step_norm = dot(&p, &p).sqrt();

        // On large-residual problems GN underestimates curvature and
        // overshoots. Fit a parabola through f(0), f'(0) and f(1) along `p`.
        let mut step_length = 1.0;
        let slope = dot(&p, &g);
        let curvature = 0.5 * (f_new - f) - slope;
        let fraction = (-slope / (2.0 * if curvature > 0.0 { curvature } else { 1.0 }))
            .clamp(s.ls_min_fraction, 1.0);
        if f_new.is_finite() && curvature > 0.0 && fraction < 0.95 {
            let p_short: Vec<f64> = p.iter().map(|v| fraction * v).collect();
            let (r_short, tape_short) = self.problem.residuals(&step_to(&p_short))?;
            let f_short = dot(&r_short, &r_short);
            // Kept only if it actually beats the full step; NaN compares false.
            if f_short < f_new {
                p = p_short;
                r_new = r_short;
                tape_new = tape_short;
                // `J` is linear, so the shortened step's `Jp` is just rescaled.
                for v in jp.iter_mut() {
                    *v *= fraction;
                }
                f_new = f_short;
                step_length = fraction;
            }
        }

        let actual = 0.5 * (f - f_new);
        let pred = -dot(&p, &g) - 0.5 * dot(&jp, &jp);
        let rho = safe_ratio(actual, pred);
        let ok = f_new.is_finite() && rho.is_finite();
        let accept = rho > s.accept_rho && pred > 0.0 && ok;

        // Nielsen's update, graded by `rho`; no further down than the
        // log-midpoint to `lam_lo`, the last `lam` that overshot. Two clean
        // full steps in a row, each predicted almost exactly by the GN model,
        // show that the overshoot is behind us, so `lam_lo` is forgotten. One
        // is not enough: right after a shortened step, the raised `lam` makes
        // the next step clean too, and forgetting `lam_lo` there drops `lam`
        // straight back to where it overshot.
        let clean = full_step_good && step_length == 1.0 && (rho_full - 1.0).abs() < s.lam_lo_reset;
        let clean_steps = if clean { self.clean_steps + 1 } else { 0 };
        let lam_lo = if clean_steps >= 2 { 0.0 } else { self.lam_lo };
        let lam_decrease = (1.0 / 3.0_f64).max(1.0 - (2.0 * rho_full - 1.0).powi(3));
        let lam_decreased = (lam * lam_decrease).max(lam.min((lam * lam_lo).sqrt()));
        let (mut lam_next, mut nu_out, mut lam_lo_out) = if full_step_good {
            (lam_decreased, s.nu0, lam_lo / s.lam_lo_decay)
        } else {
            (lam * self.nu, 2.0 * self.nu, self.lam_lo)
        };
        // After a shortened step of fraction `a`, `lam / a` makes the next
        // GN step about the length the line search found.
        if accept && step_length < 1.0 {
            lam_next = lam / step_length;
            nu_out = s.nu0;
            lam_lo_out = lam;
        }
        let lam_out = lam_next.clamp(s.lam_min, s.lam_max);

        // CG forcing term for the next step.
        let mut eta_next = match s.forcing {
            Forcing::Rho => {
                let value = (1.0 - rho_full).abs();
                if value.is_finite() {
                    value
                } else {
                    s.cg_eta_max
                }
            }
            Forcing::Residual => {
                let linear: f64 = self
                    .r
                    .iter()
                    .zip(&jp)
                    .map(|(r, j)| (r + j) * (r + j))
                    .sum::<f64>()
                    .sqrt();
                (f_new.sqrt() - linear).abs() / (if f > 0.0 { f } else { 1.0 }).sqrt()
            }
            Forcing::Model => s.cg_tol,
        };
        // Eisenstat-Walker safeguard against decreasing `eta` too fast.
        let safeguard = s.cg_gamma * eta.powf(s.cg_alpha);
        if safeguard > 0.1 {
            eta_next = eta_next.max(safeguard);
        }
        if !(accept && ok) {
            eta_next = eta;
        }

        let info = StepInfo {
            f,
            f_new,
            f_out: if accept { f_new } else { f },
            accept,
            rho,
            rho_full,
            actual,
            pred,
            lam_in: lam,
            lam_out,
            lam_nu: self.nu,
            n_cg,
            cg_eta: eta,
            cg_converged: n_cg < s.cg_max,
            grad_norm: dot(&g, &g).sqrt(),
            step_norm: dot(&p, &p).sqrt(),
            full_step_norm,
            finite: ok,
            rebuilt_blocks: rebuild,
            step_length,
            nonfinite_blocks: blocks.nonfinite,
            failed_inverses,
            lmp_rank,
        };

        if accept {
            self.theta = step_to(&p);
            self.r = r_new;
            self.tape = tape_new;
            self.p_prev = p;
            self.accepted_since_build += 1;
        } else {
            // A rejected `p` was solved for a different `lam`: discard it.
            self.p_prev.fill(0.0);
        }
        self.lam = lam_out;
        self.nu = nu_out;
        self.lam_lo = lam_lo_out;
        self.clean_steps = clean_steps;
        self.eta = eta_next;
        Ok(info)
    }
}

/// Levenberg-Marquardt fit of a `FisherResiduals`, see
/// `nutpie.triangular_lm.fit`.
#[pyclass(name = "LmOptimizer")]
pub struct PyLmOptimizer {
    inner: LmOptimizer,
}

#[pymethods]
impl PyLmOptimizer {
    #[new]
    #[pyo3(signature = (
        problem,
        theta,
        *,
        lam = 1e-1,
        cg_max = 300,
        cg_tol = 1e-2,
        cg_model_tol = 0.5,
        cg_model_cost = 15.0,
        cg_eta_max = 0.5,
        cg_gamma = 0.9,
        cg_alpha = 1.618,
        accept_rho = 0.1,
        nu0 = 2.0,
        lam_min = 1e-6,
        lam_max = 1e10,
        lam_lo_decay = 1.7320508075688772,
        lam_lo_reset = 0.25,
        ls_min_fraction = 0.1,
        forcing = "residual",
        max_block_size = 256,
        rebuild_every = 1,
        lmp_size = 0,
        lmp_tol = 1e-8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        problem: &PyFisherResiduals,
        theta: PyReadonlyArray1<'_, f64>,
        lam: f64,
        cg_max: usize,
        cg_tol: f64,
        cg_model_tol: f64,
        cg_model_cost: f64,
        cg_eta_max: f64,
        cg_gamma: f64,
        cg_alpha: f64,
        accept_rho: f64,
        nu0: f64,
        lam_min: f64,
        lam_max: f64,
        lam_lo_decay: f64,
        lam_lo_reset: f64,
        ls_min_fraction: f64,
        forcing: &str,
        max_block_size: usize,
        rebuild_every: usize,
        lmp_size: usize,
        lmp_tol: f64,
    ) -> Result<Self> {
        let forcing = match forcing {
            "residual" => Forcing::Residual,
            "rho" => Forcing::Rho,
            "model" => Forcing::Model,
            other => bail!("unknown forcing {other:?}, expected 'residual', 'rho' or 'model'"),
        };
        let settings = Settings {
            cg_max,
            cg_tol,
            cg_model_tol,
            cg_model_cost,
            cg_eta_max,
            cg_gamma,
            cg_alpha,
            accept_rho,
            nu0,
            lam_min,
            lam_max,
            lam_lo_decay,
            lam_lo_reset,
            ls_min_fraction,
            forcing,
            max_block_size,
            rebuild_every,
            lmp_size,
            lmp_tol,
        };
        let theta = theta.as_slice()?.to_vec();
        let problem = problem.inner.clone();
        let inner = py.detach(|| LmOptimizer::new(problem, theta, lam, settings))?;
        Ok(Self { inner })
    }

    /// One LM step; returns its info as a dict (see `lmopt.step`).
    fn step<'py>(&mut self, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        let info = py.detach(|| self.inner.step())?;
        Ok(pythonize::pythonize(py, &info)?)
    }

    /// The current (last accepted) parameters.
    #[getter]
    fn theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.theta.clone())
    }

    /// The current residuals, flattened ``(n_draw, n_residuals)``.
    #[getter]
    fn residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.r.clone())
    }

    /// The current loss, ``sum(r**2)``.
    #[getter]
    fn loss(&self) -> f64 {
        dot(&self.inner.r, &self.inner.r)
    }

    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lam
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `P A s = s` for every stored direction, whatever `M`, also when one
    /// direction depends on the others.
    #[test]
    fn lmp_inverts_a_on_the_stored_directions() {
        let n = 12;
        let value = |a: usize, b: usize| {
            (0.37 * (a * a) as f64 + 1.3 * b as f64 + 0.71 * (a * b) as f64).sin()
        };
        let b = Mat::from_fn(n, n, &value);
        let h = &b * b.transpose();
        let diagonal: Vec<f64> = (0..n).map(|k| 1.0 + 0.1 * k as f64).collect();
        let lam = 0.3;
        let product = |s: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|r| (0..n).map(|c| h[(r, c)] * s[c]).sum())
                .collect()
        };

        let mut directions: Vec<Vec<f64>> = (0..3)
            .map(|j| (0..n).map(|k| value(j + 20, k)).collect())
            .collect();
        let dependent = directions[0].iter().zip(&directions[1]).map(|(a, b)| a + b);
        directions.push(dependent.collect());
        let pairs: VecDeque<Pair> = directions
            .iter()
            .map(|s| Pair {
                s: s.clone(),
                hs: product(s),
            })
            .collect();

        let lmp = Lmp::new(&pairs, &diagonal, lam, 1e-8).expect("rank > 0");
        assert_eq!(lmp.rank(), 3);
        let minv = |v: &[f64]| v.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
        for s in &directions {
            let mut a_s = product(s);
            for ((a, d), s) in a_s.iter_mut().zip(&diagonal).zip(s) {
                *a += lam * d * s;
            }
            let back = lmp.apply(&a_s, minv);
            for (x, y) in back.iter().zip(s) {
                assert!((x - y).abs() < 1e-8 * (1.0 + y.abs()), "{x} != {y}");
            }
        }
    }
}
