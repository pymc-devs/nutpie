"""Rust residuals of the LM fit of a `SparseTriangularMap`
(`src/triangular/lm/`, derivation in `notes/lm_derivatives.md`).

`FisherResiduals` evaluates the residuals of `FisherLoss.residuals` for the
map's conditioners, with the flow's affine and permutation frozen, plus the
operators `lmopt` needs: ``J v``, ``J^T r``, ``J^T J v`` and the exact
Gauss-Newton blocks.

Parameters live in a flat vector with one contiguous slice per variable, no
buckets and no padding (see the Rust module), for the conditioners of
`jax_conditioners`: a constant (the JAX output bias), a linear conditioner
(the skips) and an MLP (the JAX MLP without its output bias). `pack_params`
and `unpack_params` convert from and to the JAX map. A variable without
parents has no MLP weights in Rust: its JAX MLP's output is a constant,
which `pack_params` folds into the constant and `unpack_params` zeroes.

Usage::

    tmap, y, g = map_data(flow, draws, grads)
    problem = make_residuals(tmap, y, g)
    theta = pack_params(tmap)
    r = problem.residuals(theta)            # (n_draw, n_residuals), linearizes
    Jv = problem.pushforward(v)
    g = problem.pullback(r)

    theta, lam, hist = fit(problem, theta)  # LM, `src/triangular/lm/optimizer.rs`
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "check_supported",
    "fisher_divergence",
    "fit",
    "jax_conditioners",
    "make_residuals",
    "map_data",
    "pack_params",
    "param_index",
    "unpack_params",
]


_ACTIVATIONS = ("softplus", "gelu_tanh")


def _activation(tmap):
    """The `nutpie.triangular_layout` name of the conditioners' activation."""
    from nutpie.triangular_layout import _activation_name

    return _activation_name(tmap.conditioners[0].mlp.activation)


def check_supported(tmap):
    """Raise `NotImplementedError` unless the Rust residuals support `tmap`:
    depth-1 softplus or GELU (`jax.nn.gelu`, the tanh approximation)
    conditioners with the location skip, no parent features, and a
    transformer of `Contract2`, `TangentSAS`, `AsymmetricSAS` and
    `PositiveAffine` layers."""
    from nutpie.triangular import LocationSkipMlp

    if tmap.feature_degree is not None:
        raise NotImplementedError("Parent features (feature_degree) are not supported.")
    activation = None
    for conditioner in tmap.conditioners:
        if not isinstance(conditioner, LocationSkipMlp):
            raise NotImplementedError("Conditioners need the location skip.")
        mlp = conditioner.mlp
        if len(mlp.layers) != 2:
            raise NotImplementedError("Only depth-1 conditioners are supported.")
        if activation is None:
            activation = mlp.activation
        if mlp.activation is not activation or _activation(tmap) not in _ACTIVATIONS:
            raise NotImplementedError(
                "Only softplus or jax.nn.gelu conditioners, the same in every "
                "bucket, are supported."
            )
        if any(layer.bias is None for layer in mlp.layers):
            raise NotImplementedError("Conditioners need biases.")


def _parents(tmap):
    """Per variable, its real parents in conditioner input order."""
    (dim,) = tmap.shape
    parents = [None] * dim
    for members, parent_indices in zip(tmap.bucket_members, tmap.bucket_parent_indices):
        for variable, row in zip(np.asarray(members), np.asarray(parent_indices)):
            parents[int(variable)] = row[row < dim].astype(np.int64)
    return parents


def _pack(tmap, conditioners, *, fold_roots, root_w2=None):
    """The flat parameter vector of `conditioners` (shaped like
    ``tmap.conditioners``), in the Rust layout of `jax_conditioners`. With
    `fold_roots`, a variable without parents gets its MLP's constant output
    in its constant; `root_w2` collects those variables' output weights."""
    (dim,) = tmap.shape
    location = int(tmap.conditioners[0].location_index)
    scale = tmap.conditioners[0].scale_index
    slices = [None] * dim
    for bucket, conditioner in enumerate(conditioners):
        first, second = conditioner.mlp.layers
        w1 = np.asarray(first.weight)
        b1 = np.asarray(first.bias)
        w2 = np.asarray(second.weight)
        b2 = np.asarray(second.bias)
        skip = np.asarray(conditioner.skip)
        scale_skip = (
            None
            if conditioner.scale_skip is None
            else np.asarray(conditioner.scale_skip)
        )
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        n_unit = w1.shape[1]
        for local, variable in enumerate(members):
            n_parent = int(np.sum(parent_indices[local] < dim))
            constant = b2[local]
            if n_parent == 0:
                if fold_roots and n_unit:
                    hidden = np.asarray(conditioner.mlp.activation(b1[local]))
                    constant = constant + w2[local] @ hidden
                if root_w2 is not None:
                    root_w2.append(w2[local].ravel())
            # The linear conditioner's rows in coefficient order.
            rows = [(location, skip[local, :n_parent])]
            if scale is not None:
                rows.append((int(scale), scale_skip[local, :n_parent]))
            rows.sort(key=lambda row: row[0])
            parts = [constant, *(row for _, row in rows)]
            if n_parent:
                # Real parents fill the leading input slots.
                units = np.concatenate(
                    [w1[local][:, :n_parent], b1[local][:, None], w2[local].T],
                    axis=1,
                )
                parts.append(units.ravel())
            slices[int(variable)] = np.concatenate(parts)
    return np.concatenate(slices)


def _flat_conditioners(tmap):
    import equinox as eqx
    from jax.flatten_util import ravel_pytree

    arrays, static = eqx.partition(tmap.conditioners, eqx.is_inexact_array)
    flat, unravel = ravel_pytree(arrays)
    return flat, lambda flat: eqx.combine(unravel(flat), static)


def param_index(tmap):
    """``(index, roots)``: ``pack_params(tmap) == flat[index]`` where `flat`
    is the `ravel_pytree` of the conditioners' arrays, but for the constants
    of the variables without parents, and `roots` the entries of those
    variables' MLP output weights."""
    flat, unravel = _flat_conditioners(tmap)
    labels = unravel(np.arange(flat.size, dtype=np.float64))
    roots = []
    index = _pack(tmap, labels, fold_roots=False, root_w2=roots)
    roots = np.concatenate(roots) if roots else np.zeros(0)
    return index.astype(np.int64), roots.astype(np.int64)


def pack_params(tmap):
    """The conditioner parameters of `tmap` in the Rust layout."""
    check_supported(tmap)
    return np.asarray(_pack(tmap, tmap.conditioners, fold_roots=True), dtype=np.float64)


def unpack_params(tmap, theta, index=None):
    """`tmap` with its conditioners set from `theta`, and the output weights
    of variables without parents zero. Differentiable in `theta`; padded
    parent slots keep their (inert) values. `index` is `param_index(tmap)`."""
    import equinox as eqx

    flat, unravel = _flat_conditioners(tmap)
    index, roots = param_index(tmap) if index is None else index
    conditioners = unravel(flat.at[index].set(theta).at[roots].set(0.0))
    return eqx.tree_at(lambda m: m.conditioners, tmap, conditioners)


def map_data(flow, draws, grads):
    """``(tmap, y, g)``: the `SparseTriangularMap` of a flow
    `make_flow(kind="triangular")` builds, and the draws and their gradients
    as the map sees them, ``(n_draw, n_var)`` each."""
    import jax
    from paramax import unwrap

    from nutpie.transform_adapter import _triangular_map_input

    flow = unwrap(flow)
    tmap = _triangular_map_input(flow, draws[0], grads[0], 0.0)[0]
    y, g = jax.vmap(lambda d, g: _triangular_map_input(flow, d, g, 0.0)[1:])(
        draws, grads
    )
    return tmap, np.asarray(y, dtype=np.float64), np.asarray(g, dtype=np.float64)


def jax_conditioners(
    tmap, *, mlp_ridge=0.0, skip_ridge=0.0, scale_skip_ridge=0.0, shape_ridge=0.0
):
    """The Rust conditioners of `tmap`'s (see `default_conditioners`): its
    MLP's width, activation and input squash, its skips, and the ridges."""
    from nutpie.triangular_flow import default_conditioners

    first, _ = tmap.conditioners[0].mlp.layers
    return default_conditioners(
        int(first.out_features),
        _activation(tmap),
        input_squash=tmap.input_squash,
        scale_skip=tmap.conditioners[0].scale_index is not None,
        mlp_ridge=mlp_ridge,
        skip_ridge=skip_ridge,
        scale_skip_ridge=scale_skip_ridge,
        shape_ridge=shape_ridge,
    )


def make_residuals(tmap, y=None, g=None, **ridges):
    """A `FisherResiduals` for `tmap` (unwrapped), with data `y`, `g` if
    given (see `map_data`), and the ridges of `jax_conditioners`."""
    from nutpie._lib import FisherResiduals
    from nutpie.triangular_layout import _probe_transformer, transformer_dicts

    check_supported(tmap)
    (dim,) = tmap.shape
    parents = _parents(tmap)
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    parent_index = (
        np.concatenate(parents).astype(np.int64) if dim else np.zeros(0, np.int64)
    )

    _, second = tmap.conditioners[0].mlp.layers
    n_par = int(second.out_features)
    configs = _probe_transformer(tmap.transformer_constructor, n_par)

    problem = FisherResiduals(
        parent_indptr=parent_indptr,
        parent_index=parent_index,
        transformer=transformer_dicts(configs),
        n_par=n_par,
        location_index=int(tmap.conditioners[0].location_index),
        scale_index=tmap.conditioners[0].scale_index,
        conditioners=jax_conditioners(tmap, **ridges),
    )
    if y is not None:
        problem.set_data(
            np.ascontiguousarray(y, dtype=np.float64).ravel(),
            np.ascontiguousarray(g, dtype=np.float64).ravel(),
        )
    return problem


def _describe_start(prefix, objective, train, val):
    """The row before the first step: the losses ``fit`` starts from, with
    `train` (the plain Fisher divergence) only if it differs from the
    objective."""
    return (
        f"{prefix}{'start':>5}  log obj={np.log(objective):+.2f}  "
        + ("" if train == objective else f"log F={np.log(train):+.2f}  ")
        + ("" if val is None else f"val={np.log(val):+.2f}")
    ).rstrip()


def _describe_step(i, info, prefix="", show_eta=True):
    fallbacks = "".join(
        f"  {label}: {info[key]}"
        for key, label in [
            ("nonfinite_blocks", "non-finite blocks"),
            ("failed_inverses", "diagonal inverses"),
        ]
        if info[key]
    )
    return (
        f"{prefix}{i:5d}  log obj={np.log(info['F_new']):+.2f}  "
        + (
            f"log F={np.log(info['F_new'] - info['ridge_new']):+.2f}  "
            if info["ridge_new"]
            else ""
        )
        + f"rho={info['rho']:+.2f}  "
        # f"rho_full={info['rho_full']:+.2f}  "
        f"lam={info['lam_out']:.1e}  "
        f"cg={info['n_cg']:3d}{' ' if info['cg_converged'] else '*'}"
        + (f" eta={info['cg_eta']:.3f}" if show_eta else "")
        + f"{' ' if info['rebuilt_blocks'] else '~'}  "
        f"|g|={info['grad_norm']:.2e}  "
        f"|p|={info['full_step_norm']:.2e}  "
        f"a={info['step_length']:.2f}"
        + ("" if info.get("val_F") is None else f"  val={np.log(info['val_F']):+.2f}")
        + fallbacks
        + ("" if info["accept"] else "   REJECT")
    )


def fisher_divergence(problem, theta):
    """The Fisher divergence of `problem`'s draws at `theta`: the squared norm
    of its residuals."""
    r = problem.residuals(np.asarray(theta, np.float64), linearize=False)
    return float(r @ r)


def fit(
    problem,
    theta,
    *,
    n_steps=60,
    lam0=1e-1,
    min_loss=None,
    rtol=None,
    patience=5,
    verbose=True,
    should_stop=None,
    val_problem=None,
    early_stopping=True,
    prefix="",
    **settings,
):
    """Levenberg-Marquardt fit of `problem` (a `FisherResiduals` with data)
    from `theta`, in Rust (`src/triangular/lm/optimizer.rs`).

    The step is `lmopt.step` with exact blocks, Marquardt damping and the line
    search; this loop is `lmopt.fit`'s. Stops after `n_steps`, below
    `min_loss`, when `patience` steps (rejections included) lowered the loss
    by less than a fraction `rtol`, or when `should_stop()` is true.
    `settings` go to `LmOptimizer` (``cg_max``, ``forcing``,
    ``max_block_size``, ...).

    With `val_problem` (held-out draws, same map), every step also records
    their `fisher_divergence` as ``info["val_F"]``. With `early_stopping` it
    also stops when `patience` steps lowered ``val_F`` by less than a
    fraction `rtol`, the same rule as for the training loss, and returns the
    parameters of the lowest ``val_F`` instead of the last.

    With `verbose`, prints a row for the start and one per step, and why the
    fit stopped, each line beginning with `prefix`.

    Returns ``(theta, lam, hist)``: the fitted parameters, the final damping
    and one info dict per step (all of them, also after `early_stopping`
    went back to an earlier `theta`).
    """
    from nutpie._lib import LmOptimizer

    theta = np.asarray(theta, np.float64)
    optimizer = LmOptimizer(problem, theta, lam=lam0, **settings)
    best_loss, stalled = optimizer.loss, 0
    val_F = None if val_problem is None else fisher_divergence(val_problem, theta)
    best_val, best_theta, best_step = val_F, theta, -1
    val_ref, val_stalled = val_F, 0
    if verbose:
        train_F = fisher_divergence(problem, theta)
        print(_describe_start(prefix, optimizer.loss, train_F, val_F))
    # Under `forcing="model"`, `eta` is always `cg_tol`.
    show_eta = settings.get("forcing", "residual") != "model"
    hist = []
    for i in range(n_steps):
        info = optimizer.step()
        hist.append(info)
        if should_stop is not None and should_stop():
            break
        if val_problem is not None:
            # A rejected step leaves `theta`, and so `val_F`, as it was.
            if info["accept"]:
                val_F = fisher_divergence(val_problem, optimizer.theta)
            info["val_F"] = val_F
            if val_F < best_val:
                best_val = val_F
                best_theta, best_step = np.asarray(optimizer.theta), i
            # Patience, like the training loss's, only resets on a real
            # improvement: otherwise a plateau that still creeps down keeps
            # the fit going while it only overfits.
            if val_F < val_ref * (1.0 - (rtol or 0.0)):
                val_ref, val_stalled = val_F, 0
            else:
                val_stalled += 1
        if verbose:
            print(_describe_step(i, info, prefix, show_eta))

        if info["F_out"] < best_loss * (1.0 - (rtol or 0.0)):
            best_loss, stalled = info["F_out"], 0
        else:
            stalled += 1
        if min_loss and info["F_out"] < min_loss:
            break
        if rtol is not None and stalled >= patience:
            if verbose:
                print(
                    f"{prefix}objective improved by less than {rtol:g} (relative) "
                    f"in the last {patience} steps; stopping"
                )
            break
        if early_stopping and val_problem is not None and val_stalled >= patience:
            if verbose:
                print(
                    f"{prefix}validation loss improved by less than {rtol or 0:g} "
                    f"(relative) in the last {patience} steps; stopping"
                )
            break

    if early_stopping and val_problem is not None and best_step < len(hist) - 1:
        if verbose:
            step = "the start" if best_step < 0 else f"step {best_step}"
            print(f"{prefix}back to the best validation loss, at {step}")
        return best_theta, optimizer.lam, hist
    return np.asarray(optimizer.theta), optimizer.lam, hist
