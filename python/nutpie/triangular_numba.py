"""Numba backend for `SparseTriangularMap.transform_and_log_det`.

The JAX version of that method is the one place in the flow that is genuinely
sequential: ancestral sampling has to resolve each variable before the
conditioners that read it can run. `SparseTriangularMap` recovers what
parallelism there is by grouping variables into elimination levels and scanning
over those, but that costs padding (levels are rectangular), a bucket loop
inside every scan step, and -- for the banded/dense blankets where every
variable lands in its own level -- buys nothing at all, since the number of
levels equals the dimension.

Nothing about that shape is forced by the problem; it is forced by having to
express a ragged, sequential, scalar computation in terms of dense array
operations. This backend drops that requirement: variables are visited in plain
index order (which is topological, since a parent's index is always smaller),
each conditioner is evaluated at its own true input width with no padding, and
the whole thing is one `numba.njit` loop over scalars. See
`nutpie.triangular_layout` for the flattening, and `nutpie.triangular_rust` for
the same loop with the levels handed to a thread pool.

The tradeoff is that this direction is now a *forward-only* kernel: no autodiff,
no vmap, no jit-composability with the rest of the flow. It is meant for the
call sites that only ever want the value -- the leapfrog-step use in
`transform_adapter` -- while `SparseTriangularMap.transform_and_log_det` itself
stays as the differentiable reference.

Usage::

    from nutpie.triangular_numba import compile_transform

    fn = compile_transform(sparse_triangular_map)
    y, log_det = fn(x)

Everything is computed in float64.
"""

from __future__ import annotations

import numpy as np

from nutpie.triangular_layout import TriangularLayout, extract_layout

__all__ = ["compile_transform", "NumbaTriangularTransform"]


_ACTIVATION_EXPR = {
    "gelu_tanh": (
        "0.5 * v * (1.0 + np.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))"
    ),
    "relu": "max(v, 0.0)",
    "softplus": "np.log1p(np.exp(-abs(v))) + max(v, 0.0)",
    "silu": "v / (1.0 + np.exp(-v))",
    "tanh": "np.tanh(v)",
}


def _transformer_source(specs) -> str:
    """Source for the elementwise transformer chain, indices baked in.

    Mirrors `Contract2.transform_and_log_det`, with the chain unrolled and all
    structural choices (which fields exist, the log-gamma bounds) resolved at
    code-generation time rather than per call.
    """
    lines = [
        "@numba.njit(inline='always', fastmath=True, error_model='numpy')",
        "def _transformer(p, x):",
        "    log_det = 0.0",
        "    y = x",
    ]

    def ref(spec, name):
        entry = getattr(spec, name)
        if entry is None:
            return None
        index, offset = entry
        if offset == 0.0:
            return f"p[{index}]"
        return f"(p[{index}] + {offset!r})"

    for spec in specs:
        alpha = ref(spec, "alpha")
        beta = ref(spec, "beta")
        sigma = ref(spec, "sigma")
        mu = ref(spec, "mu")
        nu = ref(spec, "nu")
        bound = spec.log_gamma_bounds

        # `log gamma` is never needed on its own -- the `exp(log_sigma -
        # log_gamma)` factor is `sigma_mod / gamma` -- so the unbounded case
        # goes straight to `gamma` through `_exp_asinh`, with no transcendental
        # at all. A bounded `log gamma` is squashed through a sigmoid, so there
        # it has to be formed explicitly and exponentiated the long way.
        if alpha is None:
            lines.append("    gamma = 1.0")
        elif bound is None:
            lines.append(f"    gamma = _exp_asinh({alpha})")
        else:
            low, high = float(bound[0]), float(bound[1])
            width = high - low
            at_zero = -low / width
            offset = float(np.log(at_zero / (1.0 - at_zero)))
            slope = width / (-low * high)
            lines.append(f"    log_gamma = np.arcsinh({alpha})")
            lines.append(
                f"    log_gamma = {low!r} + {width!r} / (1.0 + np.exp("
                f"-({slope!r} * log_gamma + {offset!r})))"
            )
            lines.append("    gamma = np.exp(log_gamma)")

        lines.append(
            "    log_delta = 0.0"
            if beta is None
            else f"    log_delta = np.arcsinh({beta})"
        )
        if sigma is None:
            lines.append("    sigma_mod = 1.0")
            lines.append("    log_sigma = 0.0")
        else:
            lines.append(f"    sigma_mod = _exp_asinh({sigma})")
            lines.append("    log_sigma = np.log(sigma_mod)")

        lines.append("    centred = y" if nu is None else f"    centred = y - {nu}")
        lines.append("    half = 0.5 * centred")
        lines.append("    u = np.arcsinh(half)")
        lines.append("    arg = gamma * u + 2.0 * log_delta")
        lines.append("    sinh_arg = np.sinh(arg)")
        lines.append("    y = 2.0 * (sigma_mod / gamma) * sinh_arg")
        if mu is not None:
            lines.append(f"    y = y + {mu}")
        lines.append(
            "    log_det += log_sigma + _log_cosh_from_sinh(sinh_arg, arg)"
            " - _log_cosh_asinh(half)"
        )

    lines.append("    return y, log_det")
    return "\n".join(lines)


def _kernel_source(activation: str) -> str:
    """The sequential sweep itself.

    One pass over variables in index order. Each conditioner is a dense little
    MLP whose weights sit in one contiguous slice of `blob`, laid out
    layer-by-layer as ``weight`` (row major, ``n_out x n_in``) followed by
    ``bias``; the first layer's ``n_in`` is the variable's true parent count, so
    there is no padding anywhere in the inner loops.

    ``error_model='numpy'`` so that an overflowing transform yields inf/nan
    exactly as the JAX version does, rather than raising out of the middle of a
    leapfrog step.
    """
    return f"""
@numba.njit(inline='always', fastmath=True, error_model='numpy')
def _act(v):
    return {activation}


# `exp(asinh(a))`, which is algebraically `a + sqrt(1 + a*a)`. `Contract2`
# writes it as an arcsinh followed by an exp because the direct form cancels
# for a << 0; the conjugate `1 / (sqrt(1 + a*a) - a)` is well conditioned
# exactly there, so branching on the sign keeps the accuracy with no
# transcendental. `1 + a*a` overflows above ~1e154, where the answer is 2|a|
# or 1/(2|a|) to full precision anyway.
@numba.njit(inline='always', fastmath=True, error_model='numpy')
def _exp_asinh(a):
    if abs(a) > 1e150:
        root = abs(a)
    else:
        root = np.sqrt(1.0 + a * a)
    if a >= 0.0:
        return a + root
    return 1.0 / (root - a)


# log(cosh(arcsinh(s))) == 0.5 * log1p(s*s), since cosh(arcsinh(s)) is
# sqrt(1 + s*s). One log1p instead of the general form's exp and log1p.
@numba.njit(inline='always', fastmath=True, error_model='numpy')
def _log_cosh_asinh(s):
    if abs(s) > 1e150:
        return np.log(abs(s))
    return 0.5 * np.log1p(s * s)


# log(cosh(v)) from an already-computed sinh(v), via cosh^2 == 1 + sinh^2.
# Above |v| ~ 300 the log1p term is exactly zero in f64 -- which is also where
# sinh(v)**2 would overflow -- so both branches are exact.
@numba.njit(inline='always', fastmath=True, error_model='numpy')
def _log_cosh_from_sinh(sinh_v, v):
    if abs(v) < 300.0:
        return 0.5 * np.log1p(sinh_v * sinh_v)
    return abs(v) - 0.6931471805599453


@numba.njit(fastmath=True, error_model='numpy', boundscheck=False)
def _kernel(x, y, parent_indptr, parent_index, blob, blob_offset,
            layer_out, buf_a, buf_b):
    n_variables = x.shape[0]
    n_layers = layer_out.shape[0]
    log_det = 0.0

    for i in range(n_variables):
        start = parent_indptr[i]
        n_in = parent_indptr[i + 1] - start
        for k in range(n_in):
            buf_a[k] = y[parent_index[start + k]]

        offset = blob_offset[i]
        for layer in range(n_layers):
            n_out = layer_out[layer]
            bias_at = offset + n_in * n_out
            # Weights are stored input-major, so this is an AXPY into n_out
            # independent accumulators rather than n_out dot products. A dot
            # product's `+=` chain is a serial dependency the compiler may not
            # reassociate; these chains are independent and vectorize.
            for o in range(n_out):
                buf_b[o] = blob[bias_at + o]
            for k in range(n_in):
                value = buf_a[k]
                row = offset + k * n_out
                for o in range(n_out):
                    buf_b[o] += blob[row + o] * value
            offset = bias_at + n_out
            if layer + 1 < n_layers:
                for o in range(n_out):
                    buf_a[o] = _act(buf_b[o])
            n_in = n_out

        value, element_log_det = _transformer(buf_b, x[i])
        y[i] = value
        log_det += element_log_det

    return log_det
"""


class NumbaTriangularTransform:
    """Compiled, forward-only `transform_and_log_det` for one map.

    Holds the flattened conditioner weights, so it is only valid for the
    parameter values it was built from; rebuild it whenever the flow is
    refitted.
    """

    __slots__ = ("_kernel", "_layout", "_buf_a", "_buf_b")

    def __init__(self, kernel, layout: TriangularLayout):
        self._kernel = kernel
        self._layout = layout
        self._buf_a = np.zeros(layout.buffer_size)
        self._buf_b = np.zeros(layout.buffer_size)

    @property
    def n_variables(self) -> int:
        return self._layout.n_variables

    def __call__(self, x, out=None):
        return self.transform_and_log_det(x, out=out)

    def transform_and_log_det(self, x, out=None):
        """``x -> (y, log|det dy/dx|)``, as numpy arrays."""
        layout = self._layout
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (layout.n_variables,):
            raise ValueError(
                f"Expected an ({layout.n_variables},) array, got {x.shape}."
            )
        if out is None:
            out = np.empty_like(x)
        log_det = self._kernel(
            x,
            out,
            layout.parent_indptr,
            layout.parent_index,
            layout.blob,
            layout.blob_offset,
            layout.layer_out,
            self._buf_a,
            self._buf_b,
        )
        return out, log_det


def compile_transform(flow_map) -> NumbaTriangularTransform:
    """Compile `flow_map.transform_and_log_det` into a numba kernel.

    `flow_map` may be a `SparseTriangularMap` or an already-extracted
    `TriangularLayout`. Raises `NotImplementedError` if the map uses a
    transformer or activation this backend does not know how to reproduce.
    """
    import numba

    layout = (
        flow_map
        if isinstance(flow_map, TriangularLayout)
        else extract_layout(flow_map)
    )

    try:
        activation = _ACTIVATION_EXPR[layout.activation]
    except KeyError:
        raise NotImplementedError(
            f"The numba backend has no expression for activation "
            f"{layout.activation!r}."
        ) from None

    source = "\n".join(
        [_kernel_source(activation), _transformer_source(layout.transformer)]
    )
    namespace = {"np": np, "numba": numba}
    exec(compile(source, "<nutpie.triangular_numba>", "exec"), namespace)

    return NumbaTriangularTransform(namespace["_kernel"], layout)
