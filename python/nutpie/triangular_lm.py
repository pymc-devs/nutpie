"""Rust residuals of the LM fit of a `SparseTriangularMap`
(`src/triangular_lm.rs`, derivation in `notes/lm_derivatives.md`).

`FisherResiduals` evaluates the residuals of `FisherLoss.residuals` for the
map's conditioners, with the flow's affine and permutation frozen, plus the
operators `lmopt` needs: ``J v``, ``J^T r``, ``J^T J v`` and the exact
Gauss-Newton blocks.

Parameters live in a flat vector with one contiguous slice per variable, no
buckets and no padding (see the Rust module). `pack_params` and
`unpack_params` convert from and to the JAX map.

Usage::

    tmap, y, g = map_data(flow, draws, grads)
    problem = make_residuals(tmap, y, g)
    theta = pack_params(tmap)
    r = problem.residuals(theta)            # (n_draw, n_residuals), records
    Jv = problem.pushforward(v)
    g = problem.pullback(r)

    theta, lam, hist = fit(problem, theta)  # LM, `src/lm_optimizer.rs`
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "check_supported",
    "fit",
    "make_residuals",
    "map_data",
    "pack_params",
    "param_index",
    "unpack_params",
]


def check_supported(tmap):
    """Raise `NotImplementedError` unless the Rust residuals support `tmap`:
    depth-1 softplus conditioners with the location skip, no parent features,
    and a `Contract2` transformer."""
    import jax

    from nutpie.triangular import LocationSkipMlp

    if tmap.feature_degree is not None:
        raise NotImplementedError("Parent features (feature_degree) are not supported.")
    for conditioner in tmap.conditioners:
        if not isinstance(conditioner, LocationSkipMlp):
            raise NotImplementedError("Conditioners need the location skip.")
        mlp = conditioner.mlp
        if len(mlp.layers) != 2:
            raise NotImplementedError("Only depth-1 conditioners are supported.")
        if mlp.activation is not jax.nn.softplus:
            raise NotImplementedError("Only softplus conditioners are supported.")
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


def _pack(tmap, conditioners):
    """The flat parameter vector of `conditioners` (shaped like
    ``tmap.conditioners``), in the Rust layout."""
    (dim,) = tmap.shape
    slices = [None] * dim
    for bucket, conditioner in enumerate(conditioners):
        first, second = conditioner.mlp.layers
        w1 = np.asarray(first.weight)
        b1 = np.asarray(first.bias)
        w2 = np.asarray(second.weight)
        b2 = np.asarray(second.bias)
        skip = np.asarray(conditioner.skip)
        members = np.asarray(tmap.bucket_members[bucket])
        parent_indices = np.asarray(tmap.bucket_parent_indices[bucket])
        for local, variable in enumerate(members):
            n_parent = int(np.sum(parent_indices[local] < dim))
            # Real parents fill the leading input slots.
            units = np.concatenate(
                [
                    w1[local][:, :n_parent],
                    b1[local][:, None],
                    w2[local].T,
                ],
                axis=1,
            )
            slices[int(variable)] = np.concatenate(
                [units.ravel(), b2[local], skip[local, :n_parent]]
            )
    return np.concatenate(slices)


def _flat_conditioners(tmap):
    import equinox as eqx
    from jax.flatten_util import ravel_pytree

    arrays, static = eqx.partition(tmap.conditioners, eqx.is_inexact_array)
    flat, unravel = ravel_pytree(arrays)
    return flat, lambda flat: eqx.combine(unravel(flat), static)


def param_index(tmap):
    """``index`` with ``pack_params(tmap) == flat[index]``, where `flat` is
    the `ravel_pytree` of the conditioners' arrays."""
    flat, unravel = _flat_conditioners(tmap)
    labels = _pack(tmap, unravel(np.arange(flat.size, dtype=np.float64)))
    return labels.astype(np.int64)


def pack_params(tmap):
    """The conditioner parameters of `tmap` in the Rust layout."""
    check_supported(tmap)
    return np.asarray(_pack(tmap, tmap.conditioners), dtype=np.float64)


def unpack_params(tmap, theta, index=None):
    """`tmap` with its conditioners set from `theta`. Differentiable in
    `theta`; padded parent slots keep their (inert) values."""
    import equinox as eqx

    flat, unravel = _flat_conditioners(tmap)
    if index is None:
        index = param_index(tmap)
    conditioners = unravel(flat.at[index].set(theta))
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


def _transformer_dicts(specs):
    out = []
    for contract in specs:
        layer = {}
        for field in ("alpha", "beta", "sigma", "mu", "nu"):
            entry = getattr(contract, field)
            layer[field] = (
                None
                if entry is None
                else {"index": int(entry[0]), "offset": float(entry[1])}
            )
        bounds = contract.log_gamma_bounds
        layer["log_gamma_bounds"] = None if bounds is None else list(bounds)
        out.append(layer)
    return out


def make_residuals(tmap, y=None, g=None, *, fisher_regularization=None):
    """A `FisherResiduals` for `tmap` (unwrapped), with data `y`, `g` if
    given (see `map_data`)."""
    from nutpie._lib import FisherResiduals
    from nutpie.triangular_layout import _probe_transformer

    check_supported(tmap)
    (dim,) = tmap.shape
    parents = _parents(tmap)
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    parent_index = (
        np.concatenate(parents).astype(np.int64) if dim else np.zeros(0, np.int64)
    )

    first, second = tmap.conditioners[0].mlp.layers
    n_unit = int(first.out_features)
    n_par = int(second.out_features)
    specs = _probe_transformer(tmap.transformer_constructor, n_par)

    problem = FisherResiduals(
        parent_indptr=parent_indptr,
        parent_index=parent_index,
        n_unit=n_unit,
        n_par=n_par,
        location_index=int(tmap.conditioners[0].location_index),
        transformer=_transformer_dicts(specs),
        fisher_regularization=fisher_regularization,
    )
    if y is not None:
        problem.set_data(
            np.ascontiguousarray(y, dtype=np.float64).ravel(),
            np.ascontiguousarray(g, dtype=np.float64).ravel(),
        )
    return problem


def _describe_step(i, info):
    fallbacks = "".join(
        f"  {label}: {info[key]}"
        for key, label in [
            ("nonfinite_blocks", "non-finite blocks"),
            ("failed_inverses", "diagonal inverses"),
        ]
        if info[key]
    )
    return (
        f"{i:3d}  log F={np.log(info['F_new']):+.2f}  "
        f"rho={info['rho']:+.2f}  "
        # f"rho_full={info['rho_full']:+.2f}  "
        f"lam={info['lam_out']:.1e}  "
        f"cg={info['n_cg']:3d}{' ' if info['cg_converged'] else '*'} "
        f"eta={info['cg_eta']:.3f}"
        f"{' ' if info['rebuilt_blocks'] else '~'}  "
        f"|g|={info['grad_norm']:.2e}  "
        f"|p|={info['full_step_norm']:.2e}  "
        f"a={info['step_length']:.2f}"
        + fallbacks
        + ("" if info["accept"] else "   REJECT")
    )


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
    **settings,
):
    """Levenberg-Marquardt fit of `problem` (a `FisherResiduals` with data)
    from `theta`, in Rust (`src/lm_optimizer.rs`).

    The step is `lmopt.step` with exact blocks, Marquardt damping and the line
    search; this loop is `lmopt.fit`'s. Stops after `n_steps`, below
    `min_loss`, when `patience` steps (rejections included) lowered the loss
    by less than a fraction `rtol`, or when `should_stop()` is true.
    `settings` go to `LmOptimizer` (``cg_max``, ``forcing``,
    ``max_block_size``, ...).

    Returns ``(theta, lam, hist)``: the fitted parameters, the final damping
    and one info dict per step.
    """
    from nutpie._lib import LmOptimizer

    optimizer = LmOptimizer(
        problem, np.asarray(theta, np.float64), lam=lam0, **settings
    )
    best_loss, stalled = optimizer.loss, 0
    hist = []
    for i in range(n_steps):
        info = optimizer.step()
        hist.append(info)
        if should_stop is not None and should_stop():
            break
        if verbose:
            print(_describe_step(i, info))

        if info["F_out"] < best_loss * (1.0 - (rtol or 0.0)):
            best_loss, stalled = info["F_out"], 0
        else:
            stalled += 1
        if min_loss and info["F_out"] < min_loss:
            break
        if rtol is not None and stalled >= patience:
            if verbose:
                print(
                    f"loss improved by less than {rtol:g} (relative) in the last "
                    f"{patience} steps; stopping"
                )
            break
    return np.asarray(optimizer.theta), optimizer.lam, hist
