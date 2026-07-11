"""Pytensor/numba transform adapter for the automatic reparametrization.

:mod:`nutpie.transform_adapter` fits general normalizing flows and needs
jax/flowjax for it. The auto-reparam flow is different: its transform is
already a pytensor graph (see :mod:`nutpie.flow_reparam`) and it has only a
handful of trainable knobs (the VIP ``h``). So the whole adapter compiles to
numba and the fit is a full-batch quasi-Newton solve:

* The per-leapfrog-step hook (``init_from_transformed_position``, which nuts-rs
  calls for *every* gradient evaluation) is one numba function computing
  ``logp(constrain(z)) + logdet`` and its gradient — tens of microseconds,
  instead of a python -> jax round trip per step.
* Fitting ``h`` is scipy's L-BFGS-B on the full window, a few dozen loss
  evaluations, instead of thousands of minibatch SGD steps.

The chain, sampler space to value space, is ``z --diag--> y --flow--> x``:
a diagonal affine (estimated in closed form from each window, as in the jax
adapter) followed by the reparametrization flow.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pytensor.tensor as pt
import scipy.optimize
from pytensor.gradient import pullback
from pytensor.graph import rewrite_graph
from pytensor.graph.replace import graph_replace, vectorize_graph

from nutpie.flow_reparam import NoFlow, build_flow_graph_from_specs, free_vars_info
from nutpie.flow_reparam import reparametrize


def _model_logp_graph(model, infos, n_dim):
    """The model's logp as a graph over nutpie's flat value vector."""
    logp = rewrite_graph(model.logp(), include=["canonicalize", "stabilize"])
    joined = pt.TensorType("float64", shape=(n_dim,))("_value_point")
    by_name = {info.name: info for info in infos}
    replacements = {}
    for rv in model.free_RVs:
        value = model.rvs_to_values[rv]
        info = by_name[value.name]
        chunk = joined[info.start_idx : info.end_idx]
        replacements[value] = chunk.reshape(tuple(info.shape)).astype(rv.dtype)
    (logp,) = graph_replace([logp], replacements)
    return joined, logp


class NumbaAutoFlow:
    """The compiled numba functions the adapter drives.

    Built once per model; the adapter instances (one per chain) hold only the
    fitted parameters (``h``, and the diagonal affine's ``loc``/``scale``).
    """

    def __init__(self, model, compiled_model, specs):
        from pymc.pytensorf import compile as compile_pymc

        n_dim = int(compiled_model.n_dim)
        infos = free_vars_info(compiled_model)
        graphs = build_flow_graph_from_specs(specs, infos, n_dim)
        (y_c, h_c), (x_of_y, ljd_c) = graphs["constrain"]
        (x_u, h_u), (y_of_x, ljd_u) = graphs["unconstrain"]

        self.n_dim = n_dim
        self.n_params = int(graphs["flow_params_vector"].type.shape[0])

        h = pt.dvector("h")
        loc = pt.dvector("loc")
        scale = pt.dvector("scale")

        # ---- sampler space -> value space --------------------------------
        z = pt.dvector("z")
        x_of_z, ljd_flow = graph_replace(
            [x_of_y, ljd_c], {y_c: loc + scale * z, h_c: h}
        )
        ljd_of_z = ljd_flow + pt.log(scale).sum()

        joined, logp_of_joined = _model_logp_graph(model, infos, n_dim)
        grad_of_joined = pt.grad(logp_of_joined, joined)

        logp_z, gx_z = graph_replace([logp_of_joined, grad_of_joined], {joined: x_of_z})
        # The transformed gradient is just d/dz of the transformed logp.
        gz = pt.grad(logp_z + ljd_of_z, z)

        # ---- value space -> sampler space --------------------------------
        x = pt.dvector("x")
        gx = pt.dvector("gx")
        y_of_x_h, _ = graph_replace([y_of_x, ljd_u], {x_u: x, h_u: h})
        z_of_x = (y_of_x_h - loc) / scale
        # logdet and transformed gradient are those of the *forward* map at the
        # point we just landed on, so the sampler's bookkeeping stays in one
        # convention (matching the jax adapter's inverse_gradient_and_val).
        x_rec, ljd_rec = graph_replace([x_of_z, ljd_of_z], {z: z_of_x})
        gz_of_x = pullback([x_rec, ljd_rec], z_of_x, [gx, pt.ones(())])

        logp_x, gx_x = graph_replace([logp_of_joined, grad_of_joined], {joined: x})
        gz_of_x_own = pullback([x_rec, ljd_rec], z_of_x, [gx_x, pt.ones(())])

        # ---- the Fisher divergence over a window of draws -----------------
        X = pt.dmatrix("X")
        GX = pt.dmatrix("GX")
        Y_b = vectorize_graph(y_of_x, {x_u: X, h_u: h})
        Z_b = (Y_b - loc) / scale
        X_rec_b = vectorize_graph(x_of_z, {z: Z_b, h: h, loc: loc, scale: scale})
        LJD_b = vectorize_graph(ljd_of_z, {z: Z_b, h: h, loc: loc, scale: scale})
        GZ_b = pullback([X_rec_b, LJD_b], Z_b, [GX, pt.ones(X.shape[0])])
        loss = pt.log(pt.sum((Z_b + GZ_b) ** 2, axis=1).mean())

        # ---- draws pushed into flow space, for the diagonal estimate ------
        X_rec_y = vectorize_graph([x_of_y, ljd_c], {y_c: Y_b, h_c: h})
        GY_b = pullback(list(X_rec_y), Y_b, [GX, pt.ones(X.shape[0])])

        def build(inputs, outputs):
            # The jitted function directly, not the pytensor Function wrapper:
            # the wrapper's input validation and storage bookkeeping is a large
            # share of the cost of a call this small, and the hot hook runs once
            # per leapfrog step. Shared variables (e.g. pm.Data) become trailing
            # arguments of the jitted signature.
            fn = compile_pymc(inputs, outputs, mode="NUMBA")
            jit_fn = fn.vm.jit_fn
            shared = [var.get_value(borrow=True) for var in fn.get_shared()]
            if not shared:
                return jit_fn

            def call(*args):
                return jit_fn(*args, *shared)

            return call

        # Called once per leapfrog step -- everything else is per window.
        self.transformed = build(
            [z, h, loc, scale], [logp_z, ljd_of_z, x_of_z, gx_z, gz]
        )
        self.inv_transform = build([x, gx, h, loc, scale], [ljd_rec, z_of_x, gz_of_x])
        self.untransformed = build(
            [x, h, loc, scale], [logp_x, ljd_rec, gx_x, z_of_x, gz_of_x_own]
        )
        self.loss_and_grad = build([X, GX, h, loc, scale], [loss, pt.grad(loss, h)])
        self.push_to_flow_space = build([X, GX, h], [Y_b, GY_b])


class NumbaTransformAdapter:
    """One per chain; nuts-rs calls these methods (see ``src/wrapper.rs``)."""

    def __init__(
        self,
        seed,
        position,
        gradient,
        chain,
        *,
        flow: NumbaAutoFlow,
        window_size=600,
        num_diag_windows=6,
        initial_skip=120,
        max_iter=200,
        verbose=False,
    ):
        self._flow = flow
        self._chain = chain
        self._window_size = window_size
        self._num_diag_windows = num_diag_windows
        self._initial_skip = initial_skip
        self._max_iter = max_iter
        self._verbose = verbose

        self._h = np.zeros(flow.n_params)
        self._loc = np.zeros(flow.n_dim)
        self._scale = np.ones(flow.n_dim)
        self.index = 0

        # h = 0 is the centred no-op, so the single draw needs no push-through.
        gradient = np.asarray(gradient, dtype="float64")
        with np.errstate(divide="ignore"):
            scale = 1 / np.sqrt(np.abs(gradient))
        if np.isfinite(scale).all():
            self._scale = np.clip(scale, 1e-8, 1e8)

    @property
    def transformation_id(self):
        return self.index

    def _fit_diag(self, positions, gradients):
        """Closed-form per-dimension affine, conditional on the current flow."""
        y, gy = self._flow.push_to_flow_space(positions, gradients, self._h)
        pos_std = np.clip(y.std(0), 1e-8, 1e8)
        grad_std = np.clip(gy.std(0), 1e-8, 1e8)
        scale = np.sqrt(pos_std / grad_std)
        loc = y.mean(0) + gy.mean(0) * scale * scale
        return loc, scale

    def _loss(self, positions, gradients, h, loc, scale):
        value, _ = self._flow.loss_and_grad(positions, gradients, h, loc, scale)
        return float(value)

    def update(self, seed, positions, gradients, logps):
        self.index += 1
        if len(positions) == 0:
            return

        positions = np.ascontiguousarray(positions, dtype="float64")
        gradients = np.ascontiguousarray(gradients, dtype="float64")

        # Early windows: only re-estimate the diagonal, as the jax adapter does.
        if self.index <= self._num_diag_windows:
            size = len(positions)
            lower = -size // 5 + 3
            if len(positions[lower:]) > 0:
                positions, gradients = positions[lower:], gradients[lower:]
            loc, scale = self._fit_diag(positions, gradients)
            if np.isfinite(loc).all() and np.isfinite(scale).all():
                self._loc, self._scale = loc, scale
            return

        # Numba dispatches on types, not shapes, so a window of any length runs
        # without recompiling -- no need to pad or truncate to fixed sizes.
        positions = positions[self._initial_skip :][-self._window_size :]
        gradients = gradients[self._initial_skip :][-self._window_size :]
        if len(positions) < 10:
            return
        if not (np.isfinite(positions).all() and np.isfinite(gradients).all()):
            return

        # Re-estimate the diagonal from this window (conditional on the current
        # flow), then fit the flow knobs with the diagonal held fixed.
        loc, scale = self._fit_diag(positions, gradients)
        if not (np.isfinite(loc).all() and np.isfinite(scale).all()):
            loc, scale = self._loc, self._scale

        old_loss = self._loss(positions, gradients, self._h, loc, scale)
        if np.isfinite(old_loss):
            self._loc, self._scale = loc, scale

        def fun(h):
            value, grad = self._flow.loss_and_grad(
                positions, gradients, np.ascontiguousarray(h), loc, scale
            )
            value = float(value)
            if not np.isfinite(value):
                # Abort the line search, not the fit: scipy backtracks.
                return np.inf, np.zeros_like(h)
            return value, np.asarray(grad, dtype="float64")

        result = scipy.optimize.minimize(
            fun,
            self._h,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self._max_iter},
        )
        new_loss = self._loss(positions, gradients, result.x, loc, scale)

        if self._verbose:
            print(
                f"Chain {self._chain} window {self.index}: "
                f"loss {old_loss:.4f} -> {new_loss:.4f} in {result.nfev} evals"
            )

        if np.isfinite(new_loss) and (not np.isfinite(old_loss) or new_loss < old_loss):
            self._h = np.ascontiguousarray(result.x)

    def init_from_transformed_position(self, transformed_position):
        logp, logdet, x, gx, gz = self._flow.transformed(
            np.ascontiguousarray(transformed_position, dtype="float64"),
            self._h,
            self._loc,
            self._scale,
        )
        return float(logp), float(logdet), x, gx, gz

    def init_from_untransformed_position(self, untransformed_position):
        logp, logdet, gx, z, gz = self._flow.untransformed(
            np.ascontiguousarray(untransformed_position, dtype="float64"),
            self._h,
            self._loc,
            self._scale,
        )
        return float(logp), float(logdet), gx, z, gz

    def inv_transform(self, position, gradient):
        logdet, z, gz = self._flow.inv_transform(
            np.ascontiguousarray(position, dtype="float64"),
            np.ascontiguousarray(gradient, dtype="float64"),
            self._h,
            self._loc,
            self._scale,
        )
        return float(logdet), z, gz


def make_numba_transform_adapter(
    *,
    numba_flow: NumbaAutoFlow,
    window_size=600,
    num_diag_windows=6,
    initial_skip=120,
    max_iter=200,
    verbose=False,
    **_ignored,
):
    return partial(
        NumbaTransformAdapter,
        flow=numba_flow,
        window_size=window_size,
        num_diag_windows=num_diag_windows,
        initial_skip=initial_skip,
        max_iter=max_iter,
        verbose=verbose,
    )


def build_auto_flow_numba(model, compiled_model):
    """Compile ``model``'s auto-reparam flow for the numba adapter.

    Returns ``None`` (with a warning) if the rewrite found nothing to
    reparametrize, mirroring :func:`nutpie.flow_reparam.build_auto_flow`.
    """
    specs = reparametrize(model)
    flowed = [s for s in specs if s.flow_cls is not NoFlow]
    if not flowed:
        warnings.warn(
            "Automatic reparametrization did not find any variables to "
            "reparametrize in this model."
        )
        return None
    chosen = ", ".join(f"{s.value.name} ({s.flow_cls.__name__})" for s in flowed)
    print(
        f"auto_reparam: reparametrizing {len(flowed)} of {len(specs)} "
        f"free variables: {chosen}"
    )
    return NumbaAutoFlow(model, compiled_model, specs)
