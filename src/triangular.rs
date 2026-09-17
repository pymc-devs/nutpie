//! Forward-only `SparseTriangularMap::transform_and_log_det`.
//!
//! Ancestral sampling through the sparse triangular flow is the one genuinely
//! sequential piece of a leapfrog step: a conditioner cannot run until the
//! variables it reads are resolved. The JAX implementation in
//! `nutpie/triangular.py` expresses that as a `lax.scan` over rectangular,
//! padded elimination levels, which is the only shape XLA can work with -- and
//! which degenerates to `dim` scan steps of pure dispatch overhead whenever the
//! blanket is banded or dense.
//!
//! Here the structure is kept ragged. `nutpie.triangular_layout` flattens the
//! map into per-variable parent lists and per-variable weight slices at their
//! true widths, and this module walks them under one of three schedules (see
//! [`Schedule`]).
//!
//! Only the value and log determinant are produced. Differentiation stays in
//! JAX, where the sparse solve in `inverse_gradient_and_val` lives.
//!
//! Note on BLAS: a level is a batch of *distinct* small matrices (every
//! variable has its own conditioner), not one shared matrix applied to many
//! vectors, so there is no GEMM here for faer or any other BLAS to accelerate.
//! The conditioners are instead evaluated with `pulp`, see [`EvalMlp`].

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{bail, Result};
use numpy::{PyArray1, PyReadonlyArray1};
use pulp::{Arch, WithSimd};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;

/// Scratch space for one conditioner evaluation. Sized for the common case
/// (parent counts and hidden widths in the tens); wider maps spill to the heap.
type Scratch = SmallVec<[f64; 64]>;

/// Variables whose parents are all resolved, waiting to be picked up.
type ReadyList = SmallVec<[u32; 8]>;

/// An `f64` that one task writes and another reads.
///
/// Relaxed accesses compile to plain loads and stores, so this costs nothing
/// over a bare `f64`; it exists so that the concurrent access is expressed in
/// the memory model rather than asserted with raw pointers. Every cell is
/// written exactly once, and the ordering that makes a read of someone else's
/// cell meaningful comes from the acquire/release pair on the dependency
/// counters, not from the cell itself.
#[repr(transparent)]
#[derive(Debug)]
struct Cell(AtomicU64);

impl Cell {
    fn zeros(n: usize) -> Vec<Cell> {
        (0..n).map(|_| Cell(AtomicU64::new(0))).collect()
    }

    #[inline(always)]
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn set(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed)
    }
}

/// How to walk the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Schedule {
    /// Plain loop over levels, one thread. No atomics, no task overhead; the
    /// right answer whenever the DAG has no width to exploit, which includes
    /// every banded or dense blanket.
    Serial,
    /// Level-synchronous: each level's members in parallel, a barrier between
    /// levels. Levels below `min_parallel_work` run inline.
    Levels,
    /// Dataflow: a variable becomes runnable the moment its last parent lands,
    /// and the fork tree is built implicitly by `rayon::join` as the sweep
    /// fans out. See [`TriangularTransform::run_dataflow`].
    Dataflow,
    /// `Serial` if no level is worth parallelizing, `Dataflow` otherwise.
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Activation {
    GeluTanh,
    Relu,
    Softplus,
    Silu,
    Tanh,
}

impl Activation {
    #[inline(always)]
    fn apply(self, v: f64) -> f64 {
        match self {
            // jax.nn.gelu's default, the tanh approximation.
            Activation::GeluTanh => {
                0.5 * v * (1.0 + (0.797_884_560_802_865_4 * (v + 0.044_715 * v * v * v)).tanh())
            }
            // `f64::max` returns the non-NaN operand; `jnp.maximum` propagates.
            Activation::Relu => {
                if v.is_nan() {
                    v
                } else {
                    v.max(0.0)
                }
            }
            Activation::Softplus => (-v.abs()).exp().ln_1p() + v.max(0.0),
            Activation::Silu => v / (1.0 + (-v).exp()),
            Activation::Tanh => v.tanh(),
        }
    }
}

/// Vectorized `exp` and `log1p`, specialized to the ranges softplus needs.
///
/// Softplus is the largest single cost in a conditioner evaluation -- two libm
/// calls per hidden unit, so 32 per variable at the default width of 16 -- and
/// it is the one part of the kernel that is already a batch of independent
/// values sitting in a contiguous buffer. pulp ships no transcendentals, so
/// they are written out here.
///
/// The domains are much narrower than a general implementation must handle,
/// which is what keeps this short: `exp` is only ever called on `(-inf, 0]`,
/// and `log1p` only on `[0, 1]`.
mod softplus {
    use pulp::Simd;

    /// `1 / ln 2`.
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    /// `ln 2` split so that `LN2_HI` has trailing zero bits and `k * LN2_HI` is
    /// exact for the `|k|` that occur here (Cody-Waite).
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    /// `0x1.8p52`. Adding this to a value with `|x| < 2^51` rounds it to an
    /// integer and leaves that integer in the low mantissa bits, so the
    /// rounding and the integer extraction come out of one add.
    const SHIFTER: f64 = 6_755_399_441_055_744.0;
    /// `exp` of anything below this is zero in f64, so clamping here costs
    /// nothing and bounds `k`.
    const MIN_EXP_ARG: f64 = -746.0;
    /// `2^k` is built by writing `k + 1023` into an exponent field, which only
    /// works while that stays positive -- and `k` reaches -1076 at
    /// `MIN_EXP_ARG`. Biasing by a further `2^SCALE` and dividing it back out
    /// afterwards keeps the constructed power normal and lets the result
    /// underflow gracefully through the denormals instead of off a cliff.
    const SCALE: u64 = 537;
    const UNSCALE: f64 = f64::from_bits((1023 - SCALE) << 52);

    /// `1/2! ..= 1/13!`. The polynomial computes `exp(r) - 1 - r`, which is at
    /// most 0.07 over the reduced range, so its own rounding error enters the
    /// result scaled down by that factor; adding the `1 + r` back at the end is
    /// where the accuracy comes from. Truncating after `r^13` leaves
    /// `|r|^14/14! < 1.1e-17` over `|r| <= ln2/2`.
    const EXP_C: [f64; 12] = [
        5.000_000_000_000_000e-1,
        1.666_666_666_666_666_6e-1,
        4.166_666_666_666_666_4e-2,
        8.333_333_333_333_333e-3,
        1.388_888_888_888_889e-3,
        1.984_126_984_126_984e-4,
        2.480_158_730_158_73e-5,
        2.755_731_922_398_589_3e-6,
        2.755_731_922_398_589e-7,
        2.505_210_838_544_172e-8,
        2.087_675_698_786_81e-9,
        1.605_904_383_682_161_3e-10,
    ];

    /// fdlibm's `__ieee754_log` minimax coefficients for
    /// `log(1+f) = 2s + s*R(s*s)`, `s = f/(2+f)`, valid to better than 2^-58
    /// for `|s| <= 0.1716`, i.e. `f` in `[-0.2929, 0.4142]`.
    const LG: [f64; 7] = [
        6.666_666_666_666_735_1e-1,
        3.999_999_999_940_941_9e-1,
        2.857_142_874_366_239_1e-1,
        2.222_219_843_214_978_4e-1,
        1.818_357_216_161_805e-1,
        1.531_383_769_920_937_3e-1,
        1.479_819_860_511_658_6e-1,
    ];

    /// `sqrt(2) - 1`, the split that keeps `|s|` inside `LG`'s range.
    const SQRT2_MINUS_1: f64 = 0.414_213_562_373_095_03;

    /// `exp(a)` for `a <= 0`.
    #[inline(always)]
    fn exp_nonpositive<S: Simd>(simd: S, a: S::f64s) -> S::f64s {
        let a = simd.max_f64s(a, simd.splat_f64s(MIN_EXP_ARG));
        let shifter = simd.splat_f64s(SHIFTER);

        // `t = round(a / ln2) + SHIFTER`, so `k_f` is the rounded quotient and
        // the bits of `t` hold that same integer offset by `SHIFTER`'s bits.
        let t = simd.mul_add_e_f64s(a, simd.splat_f64s(INV_LN2), shifter);
        let k = simd.sub_f64s(t, shifter);

        // `2^k`, built by putting `k + 1023` into the exponent field. pulp has
        // no 64-bit shift, but the shift is a multiply by `2^52`.
        // pulp names these output-first: `transmute_u64s_f64s` reads f64 bits
        // as u64, and `transmute_f64s_u64s` goes back.
        let biased = simd.add_u64s(
            simd.transmute_u64s_f64s(t),
            simd.splat_u64s((1023 + SCALE).wrapping_sub(SHIFTER.to_bits())),
        );
        let two_k = simd.transmute_f64s_u64s(simd.mul_u64s(biased, simd.splat_u64s(1 << 52)));

        // `r = a - k*ln2`, in two steps so the cancellation stays exact.
        let r = simd.negate_mul_add_e_f64s(k, simd.splat_f64s(LN2_HI), a);
        let r = simd.negate_mul_add_e_f64s(k, simd.splat_f64s(LN2_LO), r);

        let mut poly = simd.splat_f64s(EXP_C[11]);
        for coefficient in EXP_C[..11].iter().rev() {
            poly = simd.mul_add_e_f64s(poly, r, simd.splat_f64s(*coefficient));
        }
        // exp(r) = 1 + (r + r*r*poly)
        let tail = simd.mul_add_e_f64s(simd.mul_f64s(r, r), poly, r);
        let exp_r = simd.add_f64s(simd.splat_f64s(1.0), tail);

        // `a <= 0` so `k <= 0` and the scaled product cannot overflow; undoing
        // the bias last is what lets small results decay into the denormals.
        simd.mul_f64s(
            simd.mul_f64s(two_k, exp_r),
            simd.splat_f64s(UNSCALE),
        )
    }

    /// `log(1 + u)` for `u` in `[0, 1]`.
    #[inline(always)]
    fn log1p_unit<S: Simd>(simd: S, u: S::f64s) -> S::f64s {
        let one = simd.splat_f64s(1.0);
        // Halve the upper part so the reduced argument lands in the interval
        // `LG` was fitted on, and pay for it with one `ln 2` at the end.
        // `u - 1` is exact for `u >= 0.5` (Sterbenz) and loses at most one
        // ulp on the narrow strip below that.
        let upper = simd.greater_than_f64s(u, simd.splat_f64s(SQRT2_MINUS_1));
        let f = simd.select_f64s(
            upper,
            simd.mul_f64s(simd.sub_f64s(u, one), simd.splat_f64s(0.5)),
            u,
        );

        let s = simd.div_f64s(f, simd.add_f64s(simd.splat_f64s(2.0), f));
        let z = simd.mul_f64s(s, s);
        let w = simd.mul_f64s(z, z);

        let mut even = simd.splat_f64s(LG[5]);
        even = simd.mul_add_e_f64s(even, w, simd.splat_f64s(LG[3]));
        even = simd.mul_add_e_f64s(even, w, simd.splat_f64s(LG[1]));
        let mut odd = simd.splat_f64s(LG[6]);
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[4]));
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[2]));
        odd = simd.mul_add_e_f64s(odd, w, simd.splat_f64s(LG[0]));
        let poly = simd.mul_add_e_f64s(simd.mul_f64s(even, w), one, simd.mul_f64s(odd, z));

        let half_square = simd.mul_f64s(simd.splat_f64s(0.5), simd.mul_f64s(f, f));
        // log(1+f) = f - (hfsq - s*(hfsq + R))
        let log_f = simd.sub_f64s(
            f,
            simd.sub_f64s(
                half_square,
                simd.mul_f64s(s, simd.add_f64s(half_square, poly)),
            ),
        );

        let zero = simd.splat_f64s(0.0);
        let hi = simd.select_f64s(upper, simd.splat_f64s(LN2_HI), zero);
        let lo = simd.select_f64s(upper, simd.splat_f64s(LN2_LO), zero);
        simd.add_f64s(simd.add_f64s(hi, lo), log_f)
    }

    /// `log(1 + exp(v))`, in the same stable arrangement as the scalar form:
    /// `max(v, 0) + log1p(exp(-|v|))`.
    #[inline(always)]
    pub fn softplus<S: Simd>(simd: S, v: S::f64s) -> S::f64s {
        let u = exp_nonpositive(simd, simd.neg_f64s(simd.abs_f64s(v)));
        let out = simd.add_f64s(simd.max_f64s(v, simd.splat_f64s(0.0)), log1p_unit(simd, u));
        // Both `max` calls above -- the `max(v, 0)` here and the clamp inside
        // `exp_nonpositive` -- return the *other* operand when given a NaN,
        // which would turn a diverged trajectory back into a finite number.
        // `jnp.maximum` propagates instead, and so must this.
        simd.select_f64s(simd.equal_f64s(v, v), out, v)
    }
}

/// `field = conditioner_output[index] + offset`.
#[derive(Debug, Clone, Copy, Deserialize)]
struct Param {
    index: usize,
    offset: f64,
}

impl Param {
    #[inline(always)]
    fn get(self, params: &[f64]) -> f64 {
        params[self.index] + self.offset
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct Contract2Spec {
    alpha: Option<Param>,
    beta: Option<Param>,
    sigma: Option<Param>,
    mu: Option<Param>,
    nu: Option<Param>,
    log_gamma_bounds: Option<(f64, f64)>,
}

/// `_bounded_log_gamma` from `nutpie/normalizing_flow.py`, with the constants
/// derived from the bounds once instead of per call.
#[derive(Debug, Clone, Copy)]
struct LogGammaBound {
    low: f64,
    width: f64,
    slope: f64,
    offset: f64,
}

impl LogGammaBound {
    fn new(low: f64, high: f64) -> Result<Self> {
        if !(low < 0.0 && 0.0 < high) {
            bail!("log_gamma_bounds must satisfy low < 0 < high, got ({low}, {high})");
        }
        let width = high - low;
        let at_zero = -low / width;
        Ok(Self {
            low,
            width,
            slope: width / (-low * high),
            offset: (at_zero / (1.0 - at_zero)).ln(),
        })
    }

    #[inline(always)]
    fn apply(&self, unbounded: f64) -> f64 {
        self.low + self.width / (1.0 + (-(self.slope * unbounded + self.offset)).exp())
    }
}

#[derive(Debug, Clone, Copy)]
struct Contract2 {
    alpha: Option<Param>,
    beta: Option<Param>,
    sigma: Option<Param>,
    mu: Option<Param>,
    nu: Option<Param>,
    bound: Option<LogGammaBound>,
}

impl Contract2 {
    fn new(spec: Contract2Spec) -> Result<Self> {
        let bound = match spec.log_gamma_bounds {
            None => None,
            Some((low, high)) => Some(LogGammaBound::new(low, high)?),
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
}

/// `exp(asinh(a))`, which is algebraically `a + sqrt(1 + a*a)`.
///
/// `Contract2` writes this as an `asinh` followed by an `exp` because the
/// direct form cancels catastrophically for `a << 0`. Taking the conjugate,
/// `a + sqrt(1 + a*a) == 1 / (sqrt(1 + a*a) - a)`, which is well conditioned
/// exactly where the direct form is not -- so picking the branch by sign gives
/// the same accuracy with no transcendental at all. That removes two libm
/// calls per `Contract2` layer for `gamma`, and another for `sigma_mod`.
#[inline(always)]
fn exp_asinh(a: f64) -> f64 {
    // `1 + a*a` overflows above ~1e154, where the result is `2|a|` or
    // `1/(2|a|)` to full precision anyway.
    let root = if a.abs() > 1e150 {
        a.abs()
    } else {
        (1.0 + a * a).sqrt()
    };
    if a >= 0.0 {
        a + root
    } else {
        1.0 / (root - a)
    }
}

/// `log(cosh(asinh(s)))`, which is `0.5 * log1p(s*s)` since
/// `cosh(asinh(s)) == sqrt(1 + s*s)`.
///
/// The transformer only ever needs `log cosh` of an `asinh`, so this replaces
/// the general stable form -- an `exp` and a `log1p` -- with one `log1p`.
#[inline(always)]
fn log_cosh_asinh(s: f64) -> f64 {
    if s.abs() > 1e150 {
        s.abs().ln()
    } else {
        0.5 * (s * s).ln_1p()
    }
}

/// `log(cosh(v))` given `sinh(v)`, from `cosh^2 == 1 + sinh^2`.
///
/// Reusing the `sinh` the transform already computed removes the `exp` the
/// general form needs. Above `|v| ~ 300` the `log1p` term is exactly zero in
/// f64 -- which is also where `sinh(v)^2` would overflow -- so both branches
/// are exact.
#[inline(always)]
fn log_cosh_from_sinh(sinh_v: f64, v: f64) -> f64 {
    if v.abs() < 300.0 {
        0.5 * (sinh_v * sinh_v).ln_1p()
    } else {
        v.abs() - std::f64::consts::LN_2
    }
}

/// The elementwise transformer chain, mirroring
/// `Contract2::transform_and_log_det`.
///
/// Algebraically identical to the Python version, but arranged so that each
/// layer costs 6 libm calls instead of 11: `log gamma` is never needed on its
/// own (the `exp(log_sigma - log_gamma)` factor is just `sigma_mod / gamma`),
/// and both `exp(asinh(.))` and the two `log cosh`es have closed forms here.
/// See `exp_asinh`, `log_cosh_asinh` and `log_cosh_from_sinh`.
#[inline]
fn transform_element(layers: &[Contract2], params: &[f64], x: f64) -> (f64, f64) {
    let mut y = x;
    let mut log_det = 0.0;

    for layer in layers {
        let gamma = match (layer.alpha, &layer.bound) {
            (None, _) => 1.0,
            (Some(alpha), None) => exp_asinh(alpha.get(params)),
            // Bounded `log gamma` is squashed through a sigmoid, so it has to
            // be formed explicitly and exponentiated the long way.
            (Some(alpha), Some(bound)) => bound.apply(alpha.get(params).asinh()).exp(),
        };
        let log_delta = layer.beta.map_or(0.0, |p| p.get(params).asinh());
        let (sigma_mod, log_sigma) = match layer.sigma {
            None => (1.0, 0.0),
            Some(sigma) => {
                let sigma_mod = exp_asinh(sigma.get(params));
                (sigma_mod, sigma_mod.ln())
            }
        };

        let centred = match layer.nu {
            None => y,
            Some(nu) => y - nu.get(params),
        };
        let half = 0.5 * centred;
        let u = half.asinh();
        let arg = gamma * u + 2.0 * log_delta;
        let sinh_arg = arg.sinh();

        y = 2.0 * (sigma_mod / gamma) * sinh_arg;
        if let Some(mu) = layer.mu {
            y += mu.get(params);
        }
        log_det += log_sigma + log_cosh_from_sinh(sinh_arg, arg) - log_cosh_asinh(half);
    }

    (y, log_det)
}

/// One conditioner MLP, vectorized across the output width.
///
/// The weights arrive input-major -- one contiguous row of `n_out` values per
/// input, see `TriangularLayout.blob` -- so a layer is an AXPY into `n_out`
/// independent accumulators rather than `n_out` dot products. A dot product's
/// `+=` chain cannot be reassociated without fast-math, so it runs at scalar
/// add latency however wide the multiply is vectorized; an AXPY has one
/// dependency chain per accumulator lane and runs at FMA throughput.
///
/// `pulp` supplies the runtime dispatch, so this picks up FMA and the widest
/// available vectors without the whole extension module having to be built for
/// a specific target.
struct EvalMlp<'a> {
    /// This variable's slice of the weight blob.
    weights: &'a [f64],
    layer_out: &'a [usize],
    activation: Activation,
    n_in: usize,
    /// The gathered parent values on entry; reused as scratch for each hidden
    /// layer's activations.
    inputs: &'a mut [f64],
    /// Receives each layer's pre-activations, and finally the transformer
    /// parameters.
    acc: &'a mut [f64],
}

/// Number of SIMD accumulators kept live at once.
///
/// Four covers a whole 16-wide hidden layer in one block under AVX2, and gives
/// four independent FMA chains, which is roughly what it takes to keep an FMA
/// unit busy through its own latency.
const ACC_BLOCK: usize = 4;

/// `acc[..N] += splat(inputs[k]) * weights[k * stride ..][..N * lanes]`, summed
/// over every `k`, with the `N` accumulators held in registers.
///
/// `N` is a const parameter so the inner loop unrolls and LLVM can keep the
/// accumulators in registers across the whole `k` loop. Written against a
/// runtime-length slice instead, it reloads and stores each accumulator around
/// every FMA -- the generated code was `vfmadd213pd` straight out of memory
/// followed by `vmovupd` back into it -- and the store-to-load round trip, not
/// the FMA, becomes the dependency chain. That cost about 3x.
///
/// Each accumulator still sums over `k` in increasing order, exactly as a
/// scalar dot product would, so this changes the schedule and not the result.
#[inline(always)]
fn axpy_block<S: pulp::Simd, const N: usize>(
    simd: S,
    acc: &mut [S::f64s],
    weights: &[f64],
    stride: usize,
    inputs: &[f64],
) {
    let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
    let width = N * lanes;

    let mut regs = [acc[0]; N];
    for i in 1..N {
        regs[i] = acc[i];
    }

    for (k, &value) in inputs.iter().enumerate() {
        let splat = simd.splat_f64s(value);
        let (columns, _) = S::as_simd_f64s(&weights[k * stride..k * stride + width]);
        let columns: &[S::f64s; N] = columns[..N].try_into().expect("width is N vectors");
        for i in 0..N {
            regs[i] = simd.mul_add_e_f64s(splat, columns[i], regs[i]);
        }
    }

    acc[..N].copy_from_slice(&regs);
}

/// `dst = activation(src)`, vectorized where it is worth it.
///
/// Softplus and relu go through SIMD; the rest fall back to a scalar loop,
/// which for the transcendental ones is no worse than before.
#[inline(always)]
fn apply_activation<S: pulp::Simd>(
    simd: S,
    activation: Activation,
    src: &[f64],
    dst: &mut [f64],
) {
    debug_assert_eq!(src.len(), dst.len());
    match activation {
        Activation::Softplus | Activation::Relu => {
            // Equal lengths, so the two splits agree lane for lane.
            let (src_head, src_tail) = S::as_simd_f64s(src);
            let (dst_head, dst_tail) = S::as_mut_simd_f64s(dst);
            if activation == Activation::Softplus {
                for (out, &pre) in dst_head.iter_mut().zip(src_head) {
                    *out = softplus::softplus(simd, pre);
                }
            } else {
                let zero = simd.splat_f64s(0.0);
                for (out, &pre) in dst_head.iter_mut().zip(src_head) {
                    // See the NaN note in `softplus::softplus`.
                    let clamped = simd.max_f64s(pre, zero);
                    *out = simd.select_f64s(simd.equal_f64s(pre, pre), clamped, pre);
                }
            }
            for (out, &pre) in dst_tail.iter_mut().zip(src_tail) {
                *out = activation.apply(pre);
            }
        }
        other => {
            for (out, &pre) in dst.iter_mut().zip(src) {
                *out = other.apply(pre);
            }
        }
    }
}

impl<'a> WithSimd for EvalMlp<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            weights,
            layer_out,
            activation,
            mut n_in,
            inputs,
            acc,
        } = self;

        let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let n_layers = layer_out.len();
        let mut offset = 0usize;

        for (layer, &n_out) in layer_out.iter().enumerate() {
            let bias_at = offset + n_in * n_out;
            acc[..n_out].copy_from_slice(&weights[bias_at..bias_at + n_out]);
            let columns = &weights[offset..bias_at];
            let values = &inputs[..n_in];

            {
                let (acc_head, acc_tail) = S::as_mut_simd_f64s(&mut acc[..n_out]);

                // Full blocks first, then whatever is left over: a layer is
                // only a few vectors wide, so the remainder arms cover every
                // width that actually occurs (16 is one full block under AVX2,
                // and the 9-wide parameter layer is two vectors plus a scalar
                // tail).
                let mut done = 0;
                while done + ACC_BLOCK <= acc_head.len() {
                    axpy_block::<S, ACC_BLOCK>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    );
                    done += ACC_BLOCK;
                }
                match acc_head.len() - done {
                    0 => {}
                    1 => axpy_block::<S, 1>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                    2 => axpy_block::<S, 2>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                    _ => axpy_block::<S, 3>(
                        simd,
                        &mut acc_head[done..],
                        &columns[done * lanes..],
                        n_out,
                        values,
                    ),
                }

                // At most `lanes - 1` outputs, so not worth blocking.
                if !acc_tail.is_empty() {
                    let base = acc_head.len() * lanes;
                    for (k, &value) in values.iter().enumerate() {
                        let row = &columns[k * n_out + base..];
                        for (slot, &w) in acc_tail.iter_mut().zip(row) {
                            *slot = value.mul_add(w, *slot);
                        }
                    }
                }
            }

            offset = bias_at + n_out;
            if layer + 1 < n_layers {
                apply_activation(simd, activation, &acc[..n_out], &mut inputs[..n_out]);
            }
            n_in = n_out;
        }
    }
}

/// Everything the dataflow sweep shares between tasks.
struct Dataflow<'a> {
    x: &'a [f64],
    y: &'a [Cell],
    /// Per-variable log det contribution, summed in index order afterwards so
    /// that the total does not depend on how rayon happened to split the work.
    log_det: &'a [Cell],
    /// Unresolved parents remaining, per variable.
    pending: &'a [AtomicU32],
}

/// A `SparseTriangularMap` flattened for evaluation. See
/// `nutpie.triangular_layout.TriangularLayout` for the field-by-field meaning;
/// the arrays are the same ones, taken by value.
pub struct TriangularTransform {
    n_variables: usize,
    parent_indptr: Vec<usize>,
    parent_index: Vec<u32>,
    /// The reverse of the parent lists, built here: the dataflow sweep pushes
    /// work forwards along edges, so it needs each variable's children.
    child_indptr: Vec<usize>,
    child_index: Vec<u32>,
    blob: Vec<f64>,
    blob_offset: Vec<usize>,
    layer_out: Vec<usize>,
    activation: Activation,
    layers: Vec<Contract2>,
    level_ptr: Vec<usize>,
    level_vars: Vec<u32>,
    /// For [`Schedule::Levels`]: whether each level is worth handing to the
    /// thread pool. Levels are tiny far more often than not -- a banded blanket
    /// puts exactly one variable in each -- and rayon's per-call overhead would
    /// then dwarf the work several times over.
    level_parallel: Vec<bool>,
    schedule: Schedule,
    buffer_size: usize,
    num_params: usize,
    /// Detected once; `dispatch` is then a match on a feature enum, so the
    /// per-variable cost of picking a SIMD path is a predictable branch.
    arch: Arch,
}

impl TriangularTransform {
    /// One variable: gather parents, run its MLP, apply the transformer.
    ///
    /// Reads only cells belonging to variables at strictly earlier levels,
    /// which every schedule guarantees are already written.
    #[inline]
    fn eval_variable(
        &self,
        variable: usize,
        y: &[Cell],
        x: f64,
        buf_a: &mut [f64],
        buf_b: &mut [f64],
    ) -> (f64, f64) {
        let start = self.parent_indptr[variable];
        let stop = self.parent_indptr[variable + 1];
        let n_in = stop - start;
        for (slot, &parent) in buf_a[..n_in]
            .iter_mut()
            .zip(&self.parent_index[start..stop])
        {
            *slot = y[parent as usize].get();
        }

        // One dispatch per variable, covering every layer: the whole MLP runs
        // inside a single `#[target_feature]` body rather than paying for the
        // feature dispatch per layer.
        self.arch.dispatch(EvalMlp {
            weights: &self.blob[self.blob_offset[variable]..self.blob_offset[variable + 1]],
            layer_out: &self.layer_out,
            activation: self.activation,
            n_in,
            inputs: buf_a,
            acc: buf_b,
        });

        transform_element(&self.layers, &buf_b[..self.num_params], x)
    }

    /// `x -> y`, returning `log|det dy/dx|`.
    pub fn transform_and_log_det(&self, x: &[f64]) -> Result<(Vec<f64>, f64)> {
        if x.len() != self.n_variables {
            bail!(
                "expected an array of length {}, got {}",
                self.n_variables,
                x.len()
            );
        }

        let y = Cell::zeros(self.n_variables);
        let log_det = match self.effective_schedule() {
            Schedule::Serial => self.run_serial(x, &y),
            Schedule::Levels => self.run_levels(x, &y),
            Schedule::Dataflow | Schedule::Auto => self.run_dataflow_top(x, &y),
        };
        Ok((y.iter().map(Cell::get).collect(), log_det))
    }

    fn effective_schedule(&self) -> Schedule {
        match self.schedule {
            Schedule::Auto => {
                if self.level_parallel.iter().any(|p| *p) {
                    Schedule::Dataflow
                } else {
                    Schedule::Serial
                }
            }
            other => other,
        }
    }

    fn run_serial(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut buf_a = vec![0.0; self.buffer_size];
        let mut buf_b = vec![0.0; self.buffer_size];
        let mut log_det = 0.0;
        for &variable in &self.level_vars {
            let variable = variable as usize;
            let (value, element) =
                self.eval_variable(variable, y, x[variable], &mut buf_a, &mut buf_b);
            y[variable].set(value);
            log_det += element;
        }
        log_det
    }

    fn run_levels(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut buf_a = vec![0.0; self.buffer_size];
        let mut buf_b = vec![0.0; self.buffer_size];
        let mut collected: Vec<(f64, f64)> = Vec::new();
        let mut log_det = 0.0;

        for level in 0..self.level_ptr.len() - 1 {
            let members = &self.level_vars[self.level_ptr[level]..self.level_ptr[level + 1]];
            if self.level_parallel[level] {
                let buffer_size = self.buffer_size;
                members
                    .par_iter()
                    .map(|&variable| {
                        let mut a: Scratch = SmallVec::from_elem(0.0, buffer_size);
                        let mut b: Scratch = SmallVec::from_elem(0.0, buffer_size);
                        let variable = variable as usize;
                        self.eval_variable(variable, y, x[variable], &mut a, &mut b)
                    })
                    .collect_into_vec(&mut collected);
                // Scatter and sum in member order, so the total does not depend
                // on how rayon split the level.
                for (&variable, &(value, element)) in members.iter().zip(&collected) {
                    y[variable as usize].set(value);
                    log_det += element;
                }
            } else {
                for &variable in members {
                    let variable = variable as usize;
                    let (value, element) =
                        self.eval_variable(variable, y, x[variable], &mut buf_a, &mut buf_b);
                    y[variable].set(value);
                    log_det += element;
                }
            }
        }
        log_det
    }

    fn run_dataflow_top(&self, x: &[f64], y: &[Cell]) -> f64 {
        let log_det = Cell::zeros(self.n_variables);
        let pending: Vec<AtomicU32> = (0..self.n_variables)
            .map(|i| {
                AtomicU32::new((self.parent_indptr[i + 1] - self.parent_indptr[i]) as u32)
            })
            .collect();

        let roots: ReadyList = (0..self.n_variables as u32)
            .filter(|&i| self.parent_indptr[i as usize + 1] == self.parent_indptr[i as usize])
            .collect();

        self.run_dataflow(
            roots,
            &Dataflow {
                x,
                y,
                log_det: &log_det,
                pending: &pending,
            },
        );

        log_det.iter().map(Cell::get).sum()
    }

    /// The dataflow sweep.
    ///
    /// Levels are a *static* schedule: they assume the useful unit of
    /// synchronization is "everything at depth d before anything at depth
    /// d + 1". That is stronger than the DAG requires, and it costs whenever
    /// the levels are uneven -- a level containing one 200-parent conditioner
    /// and two hundred 1-parent ones makes everybody wait for the slow one,
    /// and a long thin chain running beside a wide burst is stalled at every
    /// step even though nothing it needs is missing.
    ///
    /// Here each variable carries a count of unresolved parents instead.
    /// Finishing a variable decrements its children; a child whose count
    /// reaches zero is runnable *now*, whoever happens to be holding it. The
    /// task keeps one runnable variable for itself and forks the rest with
    /// `rayon::join`, so the fork tree is built implicitly from the DAG's own
    /// shape as the sweep unfolds -- no schedule is precomputed and no barrier
    /// is imposed.
    ///
    /// Two properties fall out of this that the level schedule needed
    /// heuristics for. A DAG with no width (a banded blanket: every variable
    /// has exactly one runnable successor) never forks at all and runs as a
    /// plain loop on one thread. And `rayon::join` only reaches the thread pool
    /// when a thread is actually free to steal, so a narrow burst costs a
    /// cheap failed steal rather than a scheduling round trip.
    ///
    /// The one thing `join` cannot express directly is the recursion the user
    /// might expect -- resolving a variable by joining on its *parents*. Those
    /// are shared between children, so that recursion is over a DAG, not a
    /// tree, and would need memoized nodes with tasks blocking on each other's
    /// subgoals. Pushing forward along edges keeps it a fork-join tree, which
    /// is exactly what `join` wants.
    fn run_dataflow(&self, mut ready: ReadyList, ctx: &Dataflow<'_>) {
        let mut buf_a: Scratch = SmallVec::from_elem(0.0, self.buffer_size);
        let mut buf_b: Scratch = SmallVec::from_elem(0.0, self.buffer_size);

        // LIFO, so a task follows its own chain depth-first and keeps the
        // values it just wrote in cache.
        while let Some(variable) = ready.pop() {
            let variable = variable as usize;
            let (value, element) =
                self.eval_variable(variable, ctx.y, ctx.x[variable], &mut buf_a, &mut buf_b);
            ctx.y[variable].set(value);
            ctx.log_det[variable].set(element);

            let start = self.child_indptr[variable];
            let stop = self.child_indptr[variable + 1];
            for &child in &self.child_index[start..stop] {
                // AcqRel is what makes the gather in `eval_variable` legal: the
                // release half publishes the write of `y[variable]` just made,
                // and whichever thread sees the count reach zero acquires the
                // whole release sequence, i.e. every parent's write.
                if ctx.pending[child as usize].fetch_sub(1, Ordering::AcqRel) == 1 {
                    ready.push(child);
                }
            }

            if ready.len() > 1 {
                let split = ready.len() / 2;
                let rest: ReadyList = ready.drain(split..).collect();
                rayon::join(
                    || self.run_dataflow(rest, ctx),
                    move || self.run_dataflow(ready, ctx),
                );
                return;
            }
        }
    }
}

/// Compiled forward transform for one `SparseTriangularMap`.
///
/// Built from the arrays of `nutpie.triangular_layout.extract_layout`, which
/// bake in the conditioner weights -- so an instance is only valid for the
/// parameter values it was created from. Use
/// `nutpie.triangular_rust.compile_transform` rather than calling this
/// constructor directly.
#[pyclass(name = "SparseTriangularTransform")]
pub struct PySparseTriangularTransform {
    inner: TriangularTransform,
}

#[pymethods]
impl PySparseTriangularTransform {
    #[new]
    #[pyo3(signature = (
        *,
        parent_indptr,
        parent_index,
        blob,
        blob_offset,
        layer_out,
        activation,
        transformer,
        level_ptr,
        level_vars,
        level_work,
        min_parallel_work,
        schedule,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        parent_indptr: PyReadonlyArray1<'_, i64>,
        parent_index: PyReadonlyArray1<'_, i64>,
        blob: PyReadonlyArray1<'_, f64>,
        blob_offset: PyReadonlyArray1<'_, i64>,
        layer_out: PyReadonlyArray1<'_, i64>,
        activation: &str,
        transformer: &Bound<'_, PyAny>,
        level_ptr: PyReadonlyArray1<'_, i64>,
        level_vars: PyReadonlyArray1<'_, i64>,
        level_work: PyReadonlyArray1<'_, i64>,
        min_parallel_work: i64,
        schedule: &str,
    ) -> Result<Self> {
        let activation: Activation = from_tag(activation)
            .map_err(|_| anyhow::anyhow!("unknown activation {activation:?}"))?;
        let schedule: Schedule =
            from_tag(schedule).map_err(|_| anyhow::anyhow!("unknown schedule {schedule:?}"))?;

        let specs: Vec<Contract2Spec> = pythonize::depythonize(transformer)?;
        let layers = specs
            .into_iter()
            .map(Contract2::new)
            .collect::<Result<Vec<_>>>()?;

        let parent_indptr: Vec<usize> = as_usize(parent_indptr.as_slice()?)?;
        let parent_index: Vec<u32> = as_u32(parent_index.as_slice()?)?;
        let blob_offset: Vec<usize> = as_usize(blob_offset.as_slice()?)?;
        let layer_out: Vec<usize> = as_usize(layer_out.as_slice()?)?;
        let level_ptr: Vec<usize> = as_usize(level_ptr.as_slice()?)?;
        let level_vars: Vec<u32> = as_u32(level_vars.as_slice()?)?;
        let level_work = level_work.as_slice()?;
        let blob = blob.as_slice()?.to_vec();

        let n_variables = parent_indptr.len().saturating_sub(1);
        if blob_offset.len() != n_variables + 1 {
            bail!("blob_offset must have one more entry than there are variables");
        }
        if level_vars.len() != n_variables {
            bail!("level_vars must list every variable exactly once");
        }
        if level_ptr.len() != level_work.len() + 1 {
            bail!("level_ptr must have one more entry than level_work");
        }
        if layer_out.is_empty() {
            bail!("conditioners must have at least one layer");
        }
        if parent_index.iter().any(|&p| p as usize >= n_variables) {
            bail!("parent_index contains an out of range variable");
        }
        if level_vars.iter().any(|&v| v as usize >= n_variables) {
            bail!("level_vars contains an out of range variable");
        }

        let (child_indptr, child_index) =
            invert_edges(n_variables, &parent_indptr, &parent_index);

        let num_params = *layer_out.last().expect("checked non-empty");
        let max_parents = parent_indptr
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0);
        let buffer_size = layer_out
            .iter()
            .copied()
            .chain(std::iter::once(max_parents))
            .max()
            .unwrap_or(1)
            .max(1);

        let level_parallel = level_work
            .iter()
            .enumerate()
            .map(|(level, &work)| {
                let size = level_ptr[level + 1] - level_ptr[level];
                size > 1 && work >= min_parallel_work
            })
            .collect();

        Ok(Self {
            inner: TriangularTransform {
                n_variables,
                parent_indptr,
                parent_index,
                child_indptr,
                child_index,
                blob,
                blob_offset,
                layer_out,
                activation,
                layers,
                level_ptr,
                level_vars,
                level_parallel,
                schedule,
                buffer_size,
                num_params,
                arch: Arch::new(),
            },
        })
    }

    #[getter]
    fn n_variables(&self) -> usize {
        self.inner.n_variables
    }

    /// The schedule that will actually run, with `auto` resolved.
    #[getter]
    fn schedule(&self) -> &'static str {
        match self.inner.effective_schedule() {
            Schedule::Serial => "serial",
            Schedule::Levels => "levels",
            Schedule::Dataflow => "dataflow",
            Schedule::Auto => unreachable!("resolved by effective_schedule"),
        }
    }

    /// Number of levels the `levels` schedule would run on the thread pool.
    #[getter]
    fn n_parallel_levels(&self) -> usize {
        self.inner.level_parallel.iter().filter(|p| **p).count()
    }

    /// `x -> (y, log|det dy/dx|)`.
    fn transform_and_log_det<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> Result<(Bound<'py, PyArray1<f64>>, f64)> {
        let x = x.as_slice()?.to_vec();
        let (y, log_det) = py.detach(|| self.inner.transform_and_log_det(&x))?;
        Ok((PyArray1::from_vec(py, y), log_det))
    }
}

/// Apply one activation through the SIMD path, for accuracy testing.
///
/// The hidden pre-activations a real flow produces are all O(1), so exercising
/// the vectorized `exp`/`log1p` over their full domain needs a direct handle.
#[pyfunction]
#[pyo3(name = "_activation_for_testing")]
pub fn activation_for_testing<'py>(
    py: Python<'py>,
    activation: &str,
    x: PyReadonlyArray1<'py, f64>,
) -> Result<Bound<'py, PyArray1<f64>>> {
    let activation: Activation =
        from_tag(activation).map_err(|_| anyhow::anyhow!("unknown activation {activation:?}"))?;
    let src = x.as_slice()?.to_vec();
    let mut dst = vec![0.0; src.len()];
    Arch::new().dispatch(ApplyActivation {
        activation,
        src: &src,
        dst: &mut dst,
    });
    Ok(PyArray1::from_vec(py, dst))
}

struct ApplyActivation<'a> {
    activation: Activation,
    src: &'a [f64],
    dst: &'a mut [f64],
}

impl<'a> WithSimd for ApplyActivation<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        apply_activation(simd, self.activation, self.src, self.dst)
    }
}

/// Reverse a CSR edge list: parents-of -> children-of.
fn invert_edges(
    n_variables: usize,
    parent_indptr: &[usize],
    parent_index: &[u32],
) -> (Vec<usize>, Vec<u32>) {
    let mut counts = vec![0usize; n_variables + 1];
    for &parent in parent_index {
        counts[parent as usize + 1] += 1;
    }
    for i in 0..n_variables {
        counts[i + 1] += counts[i];
    }
    let child_indptr = counts.clone();

    let mut child_index = vec![0u32; parent_index.len()];
    let mut cursor = counts;
    for child in 0..n_variables {
        for &parent in &parent_index[parent_indptr[child]..parent_indptr[child + 1]] {
            let slot = &mut cursor[parent as usize];
            child_index[*slot] = child as u32;
            *slot += 1;
        }
    }
    (child_indptr, child_index)
}

fn from_tag<T: for<'de> Deserialize<'de>>(tag: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(tag.to_string()))
}

fn as_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| anyhow::anyhow!("negative index {v} in layout")))
        .collect()
}

fn as_u32(values: &[i64]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&v| u32::try_from(v).map_err(|_| anyhow::anyhow!("index {v} out of range")))
        .collect()
}
