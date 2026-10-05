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
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::Deserialize;
use smallvec::SmallVec;

use super::layers::scalar::{transform_element, transform_element_with_grads};
use super::layers::{Layer, LayerSpec};
use super::pattern::{Children, Offsets, Pattern, Ragged};

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

    /// The activation and its derivative. NaN propagates through both.
    #[inline(always)]
    fn apply_with_derivative(self, v: f64) -> (f64, f64) {
        match self {
            Activation::GeluTanh => {
                const C: f64 = 0.797_884_560_802_865_4;
                let inner = C * (v + 0.044_715 * v * v * v);
                let t = inner.tanh();
                let d_inner = C * (1.0 + 3.0 * 0.044_715 * v * v);
                (
                    0.5 * v * (1.0 + t),
                    0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * d_inner,
                )
            }
            Activation::Relu => {
                if v.is_nan() {
                    (v, v)
                } else if v > 0.0 {
                    (v, 1.0)
                } else {
                    (0.0, 0.0)
                }
            }
            Activation::Softplus => {
                let u = (-v.abs()).exp();
                let sigmoid = if v >= 0.0 {
                    1.0 / (1.0 + u)
                } else {
                    u / (1.0 + u)
                };
                (u.ln_1p() + v.max(0.0), sigmoid)
            }
            Activation::Silu => {
                let s = 1.0 / (1.0 + (-v).exp());
                (v * s, s + v * s * (1.0 - s))
            }
            Activation::Tanh => {
                let t = v.tanh();
                (t, 1.0 - t * t)
            }
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
        simd.mul_f64s(simd.mul_f64s(two_k, exp_r), simd.splat_f64s(UNSCALE))
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

    /// Softplus and its derivative, which is `sigmoid`.
    ///
    /// The derivative comes out of the forward pass for a select and a divide:
    /// `u = exp(-|v|)` is already computed, and `sigmoid(v)` is `1/(1 + u)` for
    /// `v >= 0` and `u/(1 + u)` below. No second `exp`.
    #[inline(always)]
    pub fn softplus_with_derivative<S: Simd>(simd: S, v: S::f64s) -> (S::f64s, S::f64s) {
        let zero = simd.splat_f64s(0.0);
        let one = simd.splat_f64s(1.0);
        let u = exp_nonpositive(simd, simd.neg_f64s(simd.abs_f64s(v)));
        let out = simd.add_f64s(simd.max_f64s(v, zero), log1p_unit(simd, u));

        let numerator = simd.select_f64s(simd.greater_than_or_equal_f64s(v, zero), one, u);
        let derivative = simd.div_f64s(numerator, simd.add_f64s(one, u));

        let finite = simd.equal_f64s(v, v);
        (
            simd.select_f64s(finite, out, v),
            simd.select_f64s(finite, derivative, v),
        )
    }
}

/// The fixed marginal map of a parent feature, `u = g(y)` and `g'(y)`, with
/// `params = (log gamma, eps, log sigma, mu, nu)`; the inverse of one
/// sinh-arcsinh layer, as `nutpie.triangular.marginal_to_normal`.
#[inline]
fn marginal_to_normal(params: &[f64; 5], y: f64) -> (f64, f64) {
    let [log_gamma, eps, log_sigma, mu, nu] = *params;
    let half = (log_gamma - log_sigma).exp() * (y - mu) / 2.0;
    let w = (half.asinh() - eps) * (-log_gamma).exp();
    let u = 2.0 * w.sinh() + nu;
    let du = (-log_sigma).exp() * w.cosh() / (1.0 + half * half).sqrt();
    (u, du)
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
    /// When recording a tape: receives `activation'(z_l)` for every hidden
    /// layer, concatenated, which is what `BackpropMlp` needs.
    act_derivs: Option<&'a mut [f64]>,
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
fn apply_activation<S: pulp::Simd>(simd: S, activation: Activation, src: &[f64], dst: &mut [f64]) {
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

/// `dst = activation(src)` and `deriv = activation'(src)`.
///
/// Only softplus takes the SIMD path here, because it is the only one whose
/// derivative falls out of the forward computation; for the rest the scalar
/// loop is what the forward-only version would have done anyway.
#[inline(always)]
fn apply_activation_with_derivative<S: pulp::Simd>(
    simd: S,
    activation: Activation,
    src: &[f64],
    dst: &mut [f64],
    deriv: &mut [f64],
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), deriv.len());
    match activation {
        Activation::Softplus => {
            let (src_head, src_tail) = S::as_simd_f64s(src);
            let (dst_head, dst_tail) = S::as_mut_simd_f64s(dst);
            let (deriv_head, deriv_tail) = S::as_mut_simd_f64s(deriv);
            for ((out, slope), &pre) in dst_head.iter_mut().zip(deriv_head.iter_mut()).zip(src_head)
            {
                let (value, derivative) = softplus::softplus_with_derivative(simd, pre);
                *out = value;
                *slope = derivative;
            }
            for ((out, slope), &pre) in dst_tail.iter_mut().zip(deriv_tail.iter_mut()).zip(src_tail)
            {
                let (value, derivative) = activation.apply_with_derivative(pre);
                *out = value;
                *slope = derivative;
            }
        }
        other => {
            for ((out, slope), &pre) in dst.iter_mut().zip(deriv.iter_mut()).zip(src) {
                let (value, derivative) = other.apply_with_derivative(pre);
                *out = value;
                *slope = derivative;
            }
        }
    }
}

/// Weight rows processed together in the backward pass.
///
/// Each row carries two accumulators (one per cotangent row), so four rows give
/// eight independent FMA chains -- enough to cover the FMA latency -- and eight
/// horizontal reductions that can overlap instead of one stalling the next.
const BACK_BLOCK: usize = 4;

/// `N` dot products against two shared vectors: `out_a[i] = rows[i]·a` and
/// `out_b[i] = rows[i]·b`, for `N` consecutive rows of length `n_out`.
///
/// The transposed weight layout that makes the forward pass an AXPY makes the
/// backward pass a reduction, which is the shape that cannot be reassociated.
/// Doing one row at a time leaves a single accumulator chain per cotangent and
/// pays a horizontal `reduce_sum` per row, so it runs at reduction *latency*:
/// measured at 0.79 cycles per multiply-add against the forward's 0.48.
///
/// Blocking fixes it the same way `axpy_block` does. `N` is a const parameter
/// so the inner loop unrolls and all `2N` accumulators stay in registers, and
/// the vector index runs *outermost* so every one of them is advanced per
/// iteration rather than each chain being drained in turn.
///
/// Each accumulator still sums its own row in the same order, so this is a
/// scheduling change and the result is unchanged bit for bit.
#[inline(always)]
fn dot2_block<S: pulp::Simd, const N: usize>(
    simd: S,
    rows: &[f64],
    n_out: usize,
    a: &[f64],
    b: &[f64],
    out_a: &mut [f64],
    out_b: &mut [f64],
) {
    let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
    let n_vec = n_out / lanes;
    let head = n_vec * lanes;

    let (a_head, a_tail) = S::as_simd_f64s(a);
    let (b_head, b_tail) = S::as_simd_f64s(b);
    let row_vecs: [&[S::f64s]; N] =
        core::array::from_fn(|i| S::as_simd_f64s(&rows[i * n_out..i * n_out + head]).0);

    let zero = simd.splat_f64s(0.0);
    let mut acc_a = [zero; N];
    let mut acc_b = [zero; N];

    for j in 0..n_vec {
        let av = a_head[j];
        let bv = b_head[j];
        for i in 0..N {
            let w = row_vecs[i][j];
            acc_a[i] = simd.mul_add_e_f64s(w, av, acc_a[i]);
            acc_b[i] = simd.mul_add_e_f64s(w, bv, acc_b[i]);
        }
    }

    for i in 0..N {
        let mut sum_a = simd.reduce_sum_f64s(acc_a[i]);
        let mut sum_b = simd.reduce_sum_f64s(acc_b[i]);
        for ((&r, &x), &y) in rows[i * n_out + head..(i + 1) * n_out]
            .iter()
            .zip(a_tail)
            .zip(b_tail)
        {
            sum_a += r * x;
            sum_b += r * y;
        }
        out_a[i] = sum_a;
        out_b[i] = sum_b;
    }
}

/// Backpropagate two cotangent rows through one conditioner MLP.
///
/// The two rows are `d y_i / d theta` and `d log_det_i / d theta` from the
/// transformer; the results are `d y_i / d y_p` and `d log_det_i / d y_p` for
/// each parent `p`, i.e. one row each of the sparse Jacobian `A` and the log
/// det coupling `C`.
struct BackpropMlp<'a> {
    weights: &'a [f64],
    layer_out: &'a [usize],
    n_in: usize,
    /// `activation'(z_l)` for every hidden layer, concatenated in forward order.
    act_derivs: &'a [f64],
    seed_y: &'a mut [f64],
    seed_l: &'a mut [f64],
    next_y: &'a mut [f64],
    next_l: &'a mut [f64],
    out_y: &'a mut [f64],
    out_l: &'a mut [f64],
}

impl<'a> WithSimd for BackpropMlp<'a> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            weights,
            layer_out,
            n_in,
            act_derivs,
            mut seed_y,
            mut seed_l,
            mut next_y,
            mut next_l,
            out_y,
            out_l,
        } = self;

        // Re-walk the layer geometry rather than carrying it from the forward
        // pass: it is a handful of integer ops against a second buffer.
        let n_layers = layer_out.len();
        let mut layers: SmallVec<[(usize, usize, usize, usize); 8]> = SmallVec::new();
        let mut offset = 0usize;
        let mut deriv_offset = 0usize;
        let mut inputs = n_in;
        for (layer, &n_out) in layer_out.iter().enumerate() {
            layers.push((offset, inputs, n_out, deriv_offset));
            offset += inputs * n_out + n_out;
            if layer + 1 < n_layers {
                deriv_offset += n_out;
            }
            inputs = n_out;
        }

        for layer in (0..n_layers).rev() {
            let (offset, n_in_l, n_out, _) = layers[layer];
            let columns = &weights[offset..offset + n_in_l * n_out];
            let (cotangent_y, cotangent_l) = (&seed_y[..n_out], &seed_l[..n_out]);

            // Full blocks, then the remainder; `BACK_BLOCK` is 4 so the arms
            // below cover every leftover width.
            let mut done = 0;
            while done + BACK_BLOCK <= n_in_l {
                dot2_block::<S, BACK_BLOCK>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                );
                done += BACK_BLOCK;
            }
            match n_in_l - done {
                0 => {}
                1 => dot2_block::<S, 1>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
                2 => dot2_block::<S, 2>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
                _ => dot2_block::<S, 3>(
                    simd,
                    &columns[done * n_out..],
                    n_out,
                    cotangent_y,
                    cotangent_l,
                    &mut next_y[done..],
                    &mut next_l[done..],
                ),
            }
            if layer > 0 {
                // The activation feeding this layer sits at the previous
                // layer's output width, which is this layer's input width.
                let deriv_at = layers[layer - 1].3;
                let slopes = &act_derivs[deriv_at..deriv_at + n_in_l];
                for (k, &slope) in slopes.iter().enumerate() {
                    next_y[k] *= slope;
                    next_l[k] *= slope;
                }
            }
            core::mem::swap(&mut seed_y, &mut next_y);
            core::mem::swap(&mut seed_l, &mut next_l);
        }

        out_y.copy_from_slice(&seed_y[..n_in]);
        out_l.copy_from_slice(&seed_l[..n_in]);
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
            mut act_derivs,
        } = self;

        let lanes = core::mem::size_of::<S::f64s>() / core::mem::size_of::<f64>();
        let n_layers = layer_out.len();
        let mut offset = 0usize;
        let mut deriv_offset = 0usize;

        for (layer, &n_out) in layer_out.iter().enumerate() {
            let bias_at = offset + n_in * n_out;
            acc[..n_out].copy_from_slice(&weights[bias_at..bias_at + n_out]);
            let columns = &weights[offset..bias_at];
            let values = &inputs[..n_in];

            // With no inputs -- a variable without parents, or a layer after a
            // zero-width one -- the output is just the bias. The blocked loop
            // below walks the *outputs*, and would slice the then empty
            // `columns` past its end from the second block on.
            if n_in > 0 {
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
                match &mut act_derivs {
                    Some(derivs) => apply_activation_with_derivative(
                        simd,
                        activation,
                        &acc[..n_out],
                        &mut inputs[..n_out],
                        &mut derivs[deriv_offset..deriv_offset + n_out],
                    ),
                    None => apply_activation(simd, activation, &acc[..n_out], &mut inputs[..n_out]),
                }
                deriv_offset += n_out;
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

/// The sparse Jacobian of the forward map, recorded as the sweep runs.
///
/// `J = dy/dx` itself is dense -- it is the inverse of a sparse triangular
/// matrix -- but it is never needed. Writing the map as
/// `y_i = f_i(x_i, y_pa(i))` gives `J = D + A J`, so `J = (I - A)^-1 D` with
/// `D` diagonal and `A` strictly lower triangular in topological order. A
/// vector-Jacobian product is then one reverse sweep over the DAG followed by a
/// scale, and both factors are exactly as sparse as the blanket.
///
/// The log det rides along in the same structure: with
/// `ld_i = log(d f_i / d x_i)`, `grad = J^T(g + C^T 1) + b`, which folds into
/// one fused sweep. See `TriangularTransform::pullback`.
///
/// Everything here is sized once from the layout and overwritten in place, so a
/// steady-state leapfrog step allocates nothing.
struct Tape {
    /// Per edge, by edge id and interleaved:
    /// `d y_child / d y_parent` then `d ld_child / d y_parent`.
    edges: Vec<Cell>,
    /// `d y_i / d x_i`, the diagonal of `J`'s numerator.
    diag: Vec<Cell>,
    /// `d ld_i / d x_i`.
    ld_dx: Vec<Cell>,
    /// Whether a forward pass has recorded into this tape since the last
    /// pullback. A mismatched pair would yield a *wrong gradient* rather than
    /// an error, which is exactly the kind of bug that shows up weeks later as
    /// a bad acceptance rate.
    filled: bool,
}

impl Tape {
    fn new(n_variables: usize, n_edges: usize) -> Self {
        Self {
            edges: Cell::zeros(2 * n_edges),
            diag: Cell::zeros(n_variables),
            ld_dx: Cell::zeros(n_variables),
            filled: false,
        }
    }
}

/// Per-task working buffers for one variable's evaluation.
///
/// Held on the stack (spilling to the heap only for unusually wide maps) so the
/// parallel schedules can make one per task without touching the allocator.
struct Scratchpad {
    inputs: Scratch,
    acc: Scratch,
    act_derivs: Scratch,
    dy_dtheta: Scratch,
    dld_dtheta: Scratch,
    seed_y: Scratch,
    seed_l: Scratch,
    next_y: Scratch,
    next_l: Scratch,
    edge_y: Scratch,
    edge_l: Scratch,
    /// With parent features: the MLP's input cotangents, `K` per parent,
    /// before they are chained to one per parent.
    feature_y: Scratch,
    feature_l: Scratch,
}

impl Scratchpad {
    fn new(transform: &TriangularTransform) -> Self {
        let width = transform.buffer_size;
        let params = transform.num_params;
        let parents = transform.max_parents.max(1);
        Self {
            inputs: SmallVec::from_elem(0.0, width),
            acc: SmallVec::from_elem(0.0, width),
            act_derivs: SmallVec::from_elem(0.0, transform.act_deriv_len.max(1)),
            dy_dtheta: SmallVec::from_elem(0.0, params),
            dld_dtheta: SmallVec::from_elem(0.0, params),
            seed_y: SmallVec::from_elem(0.0, width),
            seed_l: SmallVec::from_elem(0.0, width),
            next_y: SmallVec::from_elem(0.0, width),
            next_l: SmallVec::from_elem(0.0, width),
            edge_y: SmallVec::from_elem(0.0, parents),
            edge_l: SmallVec::from_elem(0.0, parents),
            feature_y: SmallVec::from_elem(0.0, parents * transform.feature_degree.max(1)),
            feature_l: SmallVec::from_elem(0.0, parents * transform.feature_degree.max(1)),
        }
    }
}

/// A `SparseTriangularMap` flattened for evaluation. See
/// `nutpie.triangular_layout.TriangularLayout` for the field-by-field meaning;
/// the arrays are the same ones, taken by value.
pub struct TriangularTransform {
    n_variables: usize,
    parents: Pattern<u32>,
    /// The transpose of `parents`, built here: the dataflow sweep pushes work
    /// forwards along edges, so it needs each variable's children, and the
    /// edge ids say where the tape keeps each edge's Jacobian entries.
    children: Children<u32>,
    /// Each variable's slice of the weight blob.
    weights: Ragged<f64>,
    layer_out: Vec<usize>,
    /// Linear skip from the parents to conditioner output `skip_index`, one
    /// weight per edge. All zeros when the conditioners have none.
    skip_weight: Vec<f64>,
    skip_index: usize,
    /// Parent features: with `feature_degree = K > 0` the MLPs see, for each
    /// parent, the orthonormal Hermite polynomials of degree `1..=K` of the
    /// parent through its marginal map (`marginal_to_normal` with
    /// `feature_params[parent]`). Zero: the raw parents.
    feature_degree: usize,
    feature_params: Vec<[f64; 5]>,
    /// Each variable's `K` features and their derivatives in its value,
    /// written when the variable is evaluated and read by its children.
    features: Vec<Cell>,
    feature_derivs: Vec<Cell>,
    activation: Activation,
    layers: Vec<Layer>,
    /// The variables of each level; concatenated, an evaluation order.
    levels: Ragged<u32>,
    /// For [`Schedule::Levels`]: whether each level is worth handing to the
    /// thread pool. Levels are tiny far more often than not -- a banded blanket
    /// puts exactly one variable in each -- and rayon's per-call overhead would
    /// then dwarf the work several times over.
    level_parallel: Vec<bool>,
    schedule: Schedule,
    buffer_size: usize,
    num_params: usize,
    max_parents: usize,
    /// Total width of the hidden-layer activation derivatives, which is what
    /// `BackpropMlp` reads.
    act_deriv_len: usize,
    /// Detected once; `dispatch` is then a match on a feature enum, so the
    /// per-variable cost of picking a SIMD path is a predictable branch.
    arch: Arch,
    tape: Tape,
}

impl TriangularTransform {
    /// One variable: gather parents, run its MLP, apply the transformer.
    ///
    /// Reads only cells belonging to variables at strictly earlier levels,
    /// which every schedule guarantees are already written.
    ///
    /// With `TAPE`, the same pass also records this variable's row of the
    /// sparse Jacobian. The backward pass through the conditioner runs here
    /// rather than in the reverse sweep deliberately: the weights are still in
    /// L1 from the forward pass, whereas by the time a reverse sweep reached
    /// this variable they would have to come back from memory -- a second pass
    /// over the whole weight blob, which at high parent counts is the dominant
    /// cost.
    #[inline]
    fn eval_variable<const TAPE: bool>(
        &self,
        variable: usize,
        y: &[Cell],
        x: f64,
        pad: &mut Scratchpad,
    ) -> (f64, f64) {
        let edges = self.parents.edges(variable);
        let (start, n_in) = (edges.start, edges.len());
        let parents = self.parents.parents(variable);
        // Taken before the MLP runs, which reuses `inputs` as a layer buffer.
        let skip: f64 = self.skip_weight[edges.clone()]
            .iter()
            .zip(parents)
            .map(|(weight, &parent)| weight * y[parent as usize].get())
            .sum();
        let degree = self.feature_degree;
        let n_mlp_in = if degree == 0 {
            for (slot, &parent) in pad.inputs[..n_in].iter_mut().zip(parents) {
                *slot = y[parent as usize].get();
            }
            n_in
        } else {
            for (j, &parent) in parents.iter().enumerate() {
                let source = &self.features[parent as usize * degree..][..degree];
                for (slot, cell) in pad.inputs[j * degree..][..degree].iter_mut().zip(source) {
                    *slot = cell.get();
                }
            }
            n_in * degree
        };

        let weights = self.weights.row(variable);

        // One dispatch per variable, covering every layer: the whole MLP runs
        // inside a single `#[target_feature]` body rather than paying for the
        // feature dispatch per layer.
        self.arch.dispatch(EvalMlp {
            weights,
            layer_out: &self.layer_out,
            activation: self.activation,
            n_in: n_mlp_in,
            inputs: &mut pad.inputs,
            acc: &mut pad.acc,
            act_derivs: if TAPE {
                Some(&mut pad.act_derivs)
            } else {
                None
            },
        });
        pad.acc[self.skip_index] += skip;

        if !TAPE {
            let result = transform_element(&self.layers, &pad.acc[..self.num_params], x);
            self.store_features(variable, result.0);
            return result;
        }

        let (value, log_det, dy_dx, dld_dx) = transform_element_with_grads(
            &self.layers,
            &pad.acc[..self.num_params],
            x,
            &mut pad.dy_dtheta,
            &mut pad.dld_dtheta,
        );
        self.tape.diag[variable].set(dy_dx);
        self.tape.ld_dx[variable].set(dld_dx);

        if n_in > 0 {
            pad.seed_y[..self.num_params].copy_from_slice(&pad.dy_dtheta);
            pad.seed_l[..self.num_params].copy_from_slice(&pad.dld_dtheta);
            let (out_y, out_l) = if degree == 0 {
                (&mut pad.edge_y[..n_in], &mut pad.edge_l[..n_in])
            } else {
                (&mut pad.feature_y[..n_mlp_in], &mut pad.feature_l[..n_mlp_in])
            };
            self.arch.dispatch(BackpropMlp {
                weights,
                layer_out: &self.layer_out,
                n_in: n_mlp_in,
                act_derivs: &pad.act_derivs,
                seed_y: &mut pad.seed_y,
                seed_l: &mut pad.seed_l,
                next_y: &mut pad.next_y,
                next_l: &mut pad.next_l,
                out_y,
                out_l,
            });
            if degree > 0 {
                // Chain the feature cotangents to one per parent.
                for (j, &parent) in parents.iter().enumerate() {
                    let derivs = &self.feature_derivs[parent as usize * degree..][..degree];
                    let (mut sum_y, mut sum_l) = (0.0, 0.0);
                    for (k, derivative) in derivs.iter().enumerate() {
                        let derivative = derivative.get();
                        sum_y += pad.feature_y[j * degree + k] * derivative;
                        sum_l += pad.feature_l[j * degree + k] * derivative;
                    }
                    pad.edge_y[j] = sum_y;
                    pad.edge_l[j] = sum_l;
                }
            }
            let skip_dy = pad.dy_dtheta[self.skip_index];
            let skip_dld = pad.dld_dtheta[self.skip_index];
            for (k, &weight) in self.skip_weight[edges].iter().enumerate() {
                pad.edge_y[k] += skip_dy * weight;
                pad.edge_l[k] += skip_dld * weight;
            }
            for k in 0..n_in {
                self.tape.edges[2 * (start + k)].set(pad.edge_y[k]);
                self.tape.edges[2 * (start + k) + 1].set(pad.edge_l[k]);
            }
        }

        self.store_features(variable, value);
        (value, log_det)
    }

    /// Write `variable`'s parent features, and their derivatives, for its
    /// children to read.
    #[inline]
    fn store_features(&self, variable: usize, value: f64) {
        let degree = self.feature_degree;
        if degree == 0 {
            return;
        }
        let (u, du) = marginal_to_normal(&self.feature_params[variable], value);
        let features = &self.features[variable * degree..][..degree];
        let derivs = &self.feature_derivs[variable * degree..][..degree];
        // He_k(u) / sqrt(k!) and its derivative k He_{k-1}(u) / sqrt(k!) * du.
        let (mut previous, mut current) = (1.0, u);
        let mut norm = 1.0_f64;
        for k in 1..=degree {
            if k > 1 {
                (previous, current) = (current, u * current - (k - 1) as f64 * previous);
            }
            norm *= (k as f64).sqrt();
            features[k - 1].set(current / norm);
            derivs[k - 1].set(k as f64 * previous / norm * du);
        }
    }

    /// `x -> y`, returning `log|det dy/dx|`.
    ///
    /// With `record`, also fills the tape so that `pullback` can run.
    pub fn transform_and_log_det(&mut self, x: &[f64], record: bool) -> Result<(Vec<f64>, f64)> {
        if x.len() != self.n_variables {
            bail!(
                "expected an array of length {}, got {}",
                self.n_variables,
                x.len()
            );
        }

        let y = Cell::zeros(self.n_variables);
        let log_det = if record {
            self.sweep::<true>(x, &y)
        } else {
            self.sweep::<false>(x, &y)
        };
        self.tape.filled = record;
        Ok((y.iter().map(Cell::get).collect(), log_det))
    }

    fn sweep<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        match self.effective_schedule() {
            Schedule::Serial => self.run_serial::<TAPE>(x, y),
            Schedule::Levels => self.run_levels::<TAPE>(x, y),
            Schedule::Dataflow | Schedule::Auto => self.run_dataflow_top::<TAPE>(x, y),
        }
    }

    /// Pull a cotangent on `(y, log_det)` back to one on `x`.
    ///
    /// `grad_x = J^T(grad_y + ld_bar * C^T 1) + ld_bar * b`, which is the
    /// fused reverse sweep below: each variable collects from its children,
    /// then scales by the Jacobian diagonal.
    ///
    /// Kept serial. It is two fused multiply-adds per edge against a tape a
    /// fraction the size of the weight blob, so it is memory bound and short;
    /// the expensive half of the VJP is the Jacobian construction, which the
    /// forward sweep already schedules.
    pub fn pullback(&self, grad_y: &[f64], ld_bar: f64) -> Result<Vec<f64>> {
        if !self.tape.filled {
            bail!("pullback called without a matching recorded forward pass");
        }
        if grad_y.len() != self.n_variables {
            bail!(
                "expected a cotangent of length {}, got {}",
                self.n_variables,
                grad_y.len()
            );
        }

        let mut w = vec![0.0; self.n_variables];
        let mut grad_x = vec![0.0; self.n_variables];

        for &variable in self.levels.values().iter().rev() {
            let variable = variable as usize;
            let mut total = grad_y[variable];
            for (child, edge) in self.children.entries(variable) {
                total += self.tape.edges[2 * edge].get() * w[child]
                    + ld_bar * self.tape.edges[2 * edge + 1].get();
            }
            w[variable] = total;
            grad_x[variable] =
                self.tape.diag[variable].get() * total + ld_bar * self.tape.ld_dx[variable].get();
        }

        Ok(grad_x)
    }

    fn effective_schedule(&self) -> Schedule {
        // Without threads there is no pool to hand levels to. rayon would fall
        // back to the current thread only if spawning reports `Unsupported`,
        // which emscripten does not.
        if !cfg!(feature = "parallel") {
            return Schedule::Serial;
        }
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

    fn run_serial<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut pad = Scratchpad::new(self);
        let mut log_det = 0.0;
        for &variable in self.levels.values() {
            let variable = variable as usize;
            let (value, element) = self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad);
            y[variable].set(value);
            log_det += element;
        }
        log_det
    }

    fn run_levels<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let mut pad = Scratchpad::new(self);
        let mut collected: Vec<(f64, f64)> = Vec::new();
        let mut log_det = 0.0;

        for level in 0..self.levels.n_rows() {
            let members = self.levels.row(level);
            if self.level_parallel[level] {
                members
                    .par_iter()
                    .map(|&variable| {
                        let mut pad = Scratchpad::new(self);
                        let variable = variable as usize;
                        self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad)
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
                        self.eval_variable::<TAPE>(variable, y, x[variable], &mut pad);
                    y[variable].set(value);
                    log_det += element;
                }
            }
        }
        log_det
    }

    fn run_dataflow_top<const TAPE: bool>(&self, x: &[f64], y: &[Cell]) -> f64 {
        let log_det = Cell::zeros(self.n_variables);
        let pending: Vec<AtomicU32> = (0..self.n_variables)
            .map(|i| AtomicU32::new(self.parents.n_parent(i) as u32))
            .collect();

        let roots: ReadyList = (0..self.n_variables as u32)
            .filter(|&i| self.parents.n_parent(i as usize) == 0)
            .collect();

        self.run_dataflow::<TAPE>(
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
    fn run_dataflow<const TAPE: bool>(&self, mut ready: ReadyList, ctx: &Dataflow<'_>) {
        let mut pad = Scratchpad::new(self);

        // LIFO, so a task follows its own chain depth-first and keeps the
        // values it just wrote in cache.
        while let Some(variable) = ready.pop() {
            let variable = variable as usize;
            let (value, element) =
                self.eval_variable::<TAPE>(variable, ctx.y, ctx.x[variable], &mut pad);
            ctx.y[variable].set(value);
            ctx.log_det[variable].set(element);

            for &child in self.children.children(variable) {
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
                    || self.run_dataflow::<TAPE>(rest, ctx),
                    move || self.run_dataflow::<TAPE>(ready, ctx),
                );
                return;
            }
        }
    }
}

/// Build a transform from `nutpie.triangular_layout.extract_layout`'s arrays.
/// Shared by the Python constructor and by [`FlowTransform::from_layout`].
#[allow(clippy::too_many_arguments)]
fn build_transform(
    parent_indptr: &[i64],
    parent_index: &[i64],
    blob: &[f64],
    blob_offset: &[i64],
    layer_out: &[i64],
    skip_weight: &[f64],
    skip_index: i64,
    feature_degree: i64,
    feature_params: &[f64],
    activation: &str,
    transformer: &Bound<'_, PyAny>,
    level_ptr: &[i64],
    level_vars: &[i64],
    level_work: &[i64],
    min_parallel_work: i64,
    schedule: &str,
) -> Result<TriangularTransform> {
    let activation: Activation =
        from_tag(activation).map_err(|_| anyhow::anyhow!("unknown activation {activation:?}"))?;
    let schedule: Schedule =
        from_tag(schedule).map_err(|_| anyhow::anyhow!("unknown schedule {schedule:?}"))?;

    let specs: Vec<LayerSpec> = pythonize::depythonize(transformer)?;
    let layers = specs
        .into_iter()
        .map(Layer::new)
        .collect::<Result<Vec<_>>>()?;

    let parents: Pattern<u32> = Pattern::from_i64(parent_indptr, parent_index)?;
    let weights = Ragged::new(Offsets::from_i64(blob_offset)?, blob.to_vec())?;
    let layer_out: Vec<usize> = as_usize(layer_out)?;
    let levels = Ragged::new(Offsets::from_i64(level_ptr)?, as_u32(level_vars)?)?;
    let skip_weight = skip_weight.to_vec();

    let n_variables = parents.n_var();
    if weights.n_rows() != n_variables {
        bail!("blob_offset must have one more entry than there are variables");
    }
    if levels.values().len() != n_variables {
        bail!("level_vars must list every variable exactly once");
    }
    if levels.n_rows() != level_work.len() {
        bail!("level_ptr must have one more entry than level_work");
    }
    if layer_out.is_empty() {
        bail!("conditioners must have at least one layer");
    }
    if levels.values().iter().any(|&v| v as usize >= n_variables) {
        bail!("level_vars contains an out of range variable");
    }

    let children = parents.children();

    let num_params = *layer_out.last().expect("checked non-empty");
    if skip_weight.len() != parents.n_edge() {
        bail!("skip_weight must have one entry per parent_index entry");
    }
    let skip_index = usize::try_from(skip_index)
        .ok()
        .filter(|&index| index < num_params)
        .ok_or_else(|| anyhow::anyhow!("skip_index {skip_index} is out of range"))?;
    let max_parents = parents.max_parent();
    let act_deriv_len: usize = layer_out[..layer_out.len() - 1].iter().sum();

    let feature_degree = usize::try_from(feature_degree)
        .map_err(|_| anyhow::anyhow!("feature_degree must not be negative"))?;
    let feature_params: Vec<[f64; 5]> = if feature_degree == 0 {
        vec![[0.0; 5]; n_variables]
    } else {
        if feature_params.len() != 5 * n_variables {
            bail!("feature_params must have five entries per variable");
        }
        feature_params
            .chunks_exact(5)
            .map(|chunk| chunk.try_into().expect("chunks of five"))
            .collect()
    };

    let buffer_size = layer_out
        .iter()
        .copied()
        .chain(std::iter::once(max_parents * feature_degree.max(1)))
        .max()
        .unwrap_or(1)
        .max(1);

    let level_parallel = level_work
        .iter()
        .enumerate()
        .map(|(level, &work)| levels.row(level).len() > 1 && work >= min_parallel_work)
        .collect();

    let n_edges = parents.n_edge();
    Ok(TriangularTransform {
        n_variables,
        parents,
        children,
        weights,
        layer_out,
        skip_weight,
        skip_index,
        feature_degree,
        feature_params,
        features: Cell::zeros(n_variables * feature_degree),
        feature_derivs: Cell::zeros(n_variables * feature_degree),
        activation,
        layers,
        levels,
        level_parallel,
        schedule,
        buffer_size,
        num_params,
        max_parents,
        act_deriv_len,
        arch: Arch::new(),
        tape: Tape::new(n_variables, n_edges),
    })
}

/// The whole flow `make_flow(kind="triangular")` builds, for the leapfrog
/// step: `z -> y` with its log determinant, and the pullback of a gradient in
/// `y` back to `z`.
///
/// That flow is `Chain([Sandwich(map, Permute(p)), Affine(loc, scale)])`, and
/// `Sandwich` is `Chain([outer, inner, Invert(outer)])`, so
///
/// ```text
/// u = z[p],   v = map(u),   w = v[p^-1],   y = loc + scale * w
/// ```
///
/// with log determinant `log_det(map) + sum(log |scale|)`. Built from the dict
/// `nutpie.triangular_rust.flow_transform_layout` returns; like
/// [`TriangularTransform`], only valid for the parameters it was built from.
pub struct FlowTransform {
    /// `None` for the diagonal-only flow of the early windows, `Affine`
    /// alone: then `map` and the permutation are the identity.
    map: Option<TriangularTransform>,
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
    loc: Vec<f64>,
    scale: Vec<f64>,
    log_det_affine: f64,
    buffer: Vec<f64>,
}

impl FlowTransform {
    pub fn from_layout(layout: &Bound<'_, PyDict>) -> Result<Self> {
        fn item<'py>(layout: &Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>> {
            layout
                .get_item(key)?
                .ok_or_else(|| anyhow::anyhow!("flow transform layout is missing {key:?}"))
        }
        fn ints(layout: &Bound<'_, PyDict>, key: &str) -> Result<Vec<i64>> {
            Ok(item(layout, key)?
                .extract::<PyReadonlyArray1<i64>>()
                .map_err(PyErr::from)?
                .as_slice()?
                .to_vec())
        }
        fn floats(layout: &Bound<'_, PyDict>, key: &str) -> Result<Vec<f64>> {
            Ok(item(layout, key)?
                .extract::<PyReadonlyArray1<f64>>()
                .map_err(PyErr::from)?
                .as_slice()?
                .to_vec())
        }

        let has_map = layout.contains("parent_indptr")?;
        let map = has_map
            .then(|| {
                build_transform(
                    &ints(layout, "parent_indptr")?,
                    &ints(layout, "parent_index")?,
                    &floats(layout, "blob")?,
                    &ints(layout, "blob_offset")?,
                    &ints(layout, "layer_out")?,
                    &floats(layout, "skip_weight")?,
                    item(layout, "skip_index")?.extract()?,
                    item(layout, "feature_degree")?.extract()?,
                    &floats(layout, "feature_params")?,
                    &item(layout, "activation")?.extract::<String>()?,
                    &item(layout, "transformer")?,
                    &ints(layout, "level_ptr")?,
                    &ints(layout, "level_vars")?,
                    &ints(layout, "level_work")?,
                    item(layout, "min_parallel_work")?.extract()?,
                    &item(layout, "schedule")?.extract::<String>()?,
                )
            })
            .transpose()?;

        let loc = floats(layout, "loc")?;
        let scale = floats(layout, "scale")?;
        let n = loc.len();
        let permutation = if has_map {
            as_usize(&ints(layout, "permutation")?)?
        } else {
            (0..n).collect()
        };
        if map.as_ref().is_some_and(|map| map.n_variables != n) {
            bail!("loc must have one entry per variable of the map");
        }
        if permutation.len() != n || loc.len() != n || scale.len() != n {
            bail!("permutation, loc and scale must each have one entry per variable");
        }
        let mut inverse_permutation = vec![usize::MAX; n];
        for (position, &variable) in permutation.iter().enumerate() {
            if variable >= n || inverse_permutation[variable] != usize::MAX {
                bail!("permutation is not a permutation of 0..{n}");
            }
            inverse_permutation[variable] = position;
        }
        let log_det_affine = scale.iter().map(|s| s.abs().ln()).sum();

        Ok(Self {
            map,
            permutation,
            inverse_permutation,
            loc,
            scale,
            log_det_affine,
            buffer: vec![0.0; n],
        })
    }

    pub fn n_variables(&self) -> usize {
        self.loc.len()
    }

    /// `z -> y`, returning the log determinant. Records the tape `pullback`
    /// needs.
    pub fn transform_and_log_det(&mut self, z: &[f64], y: &mut [f64]) -> Result<f64> {
        for (slot, &source) in self.buffer.iter_mut().zip(&self.permutation) {
            *slot = z[source];
        }
        let Some(map) = self.map.as_mut() else {
            for (k, out) in y.iter_mut().enumerate() {
                *out = self.loc[k] + self.scale[k] * self.buffer[k];
            }
            return Ok(self.log_det_affine);
        };
        let (v, log_det) = map.transform_and_log_det(&self.buffer, true)?;
        for (k, out) in y.iter_mut().enumerate() {
            *out = self.loc[k] + self.scale[k] * v[self.inverse_permutation[k]];
        }
        Ok(log_det + self.log_det_affine)
    }

    /// Pull `grad_y` (with a unit log determinant cotangent) back to `z`,
    /// through the point the last `transform_and_log_det` recorded.
    pub fn pullback(&mut self, grad_y: &[f64], grad_z: &mut [f64]) -> Result<()> {
        // `w = v[p^-1]` gathers, so its pullback scatters back: `g_v = g_w[p]`.
        for (slot, &source) in self.buffer.iter_mut().zip(&self.permutation) {
            *slot = grad_y[source] * self.scale[source];
        }
        let Some(map) = self.map.as_ref() else {
            grad_z.copy_from_slice(&self.buffer);
            return Ok(());
        };
        let grad_u = map.pullback(&self.buffer, 1.0)?;
        // `u = z[p]`, likewise.
        for (k, &target) in self.permutation.iter().enumerate() {
            grad_z[target] = grad_u[k];
        }
        Ok(())
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
        skip_weight,
        skip_index,
        feature_degree,
        feature_params,
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
        skip_weight: PyReadonlyArray1<'_, f64>,
        skip_index: i64,
        feature_degree: i64,
        feature_params: PyReadonlyArray1<'_, f64>,
        activation: &str,
        transformer: &Bound<'_, PyAny>,
        level_ptr: PyReadonlyArray1<'_, i64>,
        level_vars: PyReadonlyArray1<'_, i64>,
        level_work: PyReadonlyArray1<'_, i64>,
        min_parallel_work: i64,
        schedule: &str,
    ) -> Result<Self> {
        Ok(Self {
            inner: build_transform(
                parent_indptr.as_slice()?,
                parent_index.as_slice()?,
                blob.as_slice()?,
                blob_offset.as_slice()?,
                layer_out.as_slice()?,
                skip_weight.as_slice()?,
                skip_index,
                feature_degree,
                feature_params.as_slice()?,
                activation,
                transformer,
                level_ptr.as_slice()?,
                level_vars.as_slice()?,
                level_work.as_slice()?,
                min_parallel_work,
                schedule,
            )?,
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
    ///
    /// With `record=True` the sparse Jacobian is kept so that `pullback` can
    /// run; the tape lives in this object, so a given instance supports one
    /// forward/pullback pair at a time.
    #[pyo3(signature = (x, record = false))]
    fn transform_and_log_det<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        record: bool,
    ) -> Result<(Bound<'py, PyArray1<f64>>, f64)> {
        let x = x.as_slice()?.to_vec();
        let (y, log_det) = py.detach(|| self.inner.transform_and_log_det(&x, record))?;
        Ok((PyArray1::from_vec(py, y), log_det))
    }

    /// Pull a cotangent on `(y, log_det)` back to one on `x`.
    ///
    /// Must follow a `transform_and_log_det(..., record=True)` on the same
    /// object. `log_det_bar` is the cotangent on the scalar output, which the
    /// sampler always sets to 1.
    #[pyo3(signature = (grad_y, log_det_bar = 1.0))]
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        grad_y: PyReadonlyArray1<'py, f64>,
        log_det_bar: f64,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let grad_y = grad_y.as_slice()?.to_vec();
        let grad_x = py.detach(|| self.inner.pullback(&grad_y, log_det_bar))?;
        Ok(PyArray1::from_vec(py, grad_x))
    }
}

/// The sampler's native flow transform, exposed for testing it against the
/// JAX bijection. Built from `nutpie.triangular_rust.flow_transform_layout`.
#[pyclass(name = "FlowTransform")]
pub struct PyFlowTransform {
    inner: FlowTransform,
}

#[pymethods]
impl PyFlowTransform {
    #[new]
    fn new(layout: &Bound<'_, PyDict>) -> Result<Self> {
        Ok(Self {
            inner: FlowTransform::from_layout(layout)?,
        })
    }

    /// `z -> (y, log_det)`, recording the tape `pullback` needs.
    fn transform_and_log_det<'py>(
        &mut self,
        py: Python<'py>,
        z: PyReadonlyArray1<'py, f64>,
    ) -> Result<(Bound<'py, PyArray1<f64>>, f64)> {
        let z = z.as_slice()?.to_vec();
        let mut y = vec![0.0; z.len()];
        let log_det = py.detach(|| self.inner.transform_and_log_det(&z, &mut y))?;
        Ok((PyArray1::from_vec(py, y), log_det))
    }

    /// Pull `grad_y` (with a unit log determinant cotangent) back to `z`.
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        grad_y: PyReadonlyArray1<'py, f64>,
    ) -> Result<Bound<'py, PyArray1<f64>>> {
        let grad_y = grad_y.as_slice()?.to_vec();
        let mut grad_z = vec![0.0; grad_y.len()];
        py.detach(|| self.inner.pullback(&grad_y, &mut grad_z))?;
        Ok(PyArray1::from_vec(py, grad_z))
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
