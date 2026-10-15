"""Triangular flows built in Rust (`src/triangular/flow.rs`), without JAX.

`build_triangular_flow` creates a flow: the diagonal affine from the draws,
the sparse triangular map in the given order, its transformer and each
variable's conditioners (see `default_conditioners`, and
`src/triangular/conditioner/`). The result is a `TriangularFlow`, which fits
with `nutpie.triangular_lm.fit` on its ``residuals``, maps draws back with
``inv_transform`` and gives the sampler its native ``flow_transform``.

With `default_conditioners`, the flow works like the one
`make_flow(kind="triangular")` builds in JAX, but its MLPs have no output
bias (the constant conditioner is the only one) and no weights for a
variable without parents.

`from_jax` converts a JAX flow, e.g. to compare the two.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = [
    "build_triangular_flow",
    "default_conditioners",
    "diag_affine",
    "diag_from_gradients",
    "from_jax",
    "linear_conditioners",
    "transformer_kind",
    "transformer_layout",
]


_ACTIVATIONS = {
    "softplus": "softplus",
    "gelu": "gelu_tanh",
    "gelu_tanh": "gelu_tanh",
}


def _activation_name(activation) -> str:
    """The Rust name of `activation`: a name, or a function such as
    `jax.nn.softplus` (matched by its name, without importing JAX)."""
    name = activation if isinstance(activation, str) else activation.__name__
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported activation {activation!r}, expected softplus or gelu."
        ) from None


RUST_TRANSFORMERS = ("affine", "sas", "tangent_sas", "asymmetric_sas")


def transformer_kind(
    transformer="sas",
    transformer_layers=1,
    *,
    tangent_sas_fix_b=False,
    log_gamma_bounds=None,
):
    """The transformer as the Rust flow takes it, ``{"kind": ...}``, from
    `make_transformer`'s arguments. `transformer_layout` gives its layout."""
    layers = int(transformer_layers)
    if transformer not in RUST_TRANSFORMERS:
        raise NotImplementedError(
            f"the Rust flow does not support the {transformer!r} transformer, "
            f"only {RUST_TRANSFORMERS}."
        )
    if transformer == "affine":
        if layers != 1:
            raise ValueError("an affine transformer has exactly one layer.")
        return {"kind": "affine"}
    kind = {"kind": transformer, "layers": layers}
    if transformer == "tangent_sas":
        kind["fix_b"] = bool(tangent_sas_fix_b)
    elif log_gamma_bounds is not None:
        kind["log_gamma_bounds"] = [float(b) for b in log_gamma_bounds]
    return kind


def transformer_layout(kind):
    """The layout of a transformer kind (see `transformer_kind`): its
    ``n_coefs`` coefficients, the ``location``, ``scale`` and ``shape``
    ones among them, and its ``layers`` as
    `nutpie.triangular_layout.transformer_dicts` writes them."""
    from nutpie._lib import transformer_layout as rust_layout

    return rust_layout(kind)


def diag_from_gradients(positions, gradients):
    """``(mean, scale)`` of a diagonal normal from the gradients alone, as
    `normalizing_flow.diag_from_gradients`."""
    with np.errstate(divide="ignore"):
        diag = 1 / np.sqrt(np.abs(gradients).mean(0))
    diag = np.clip(np.where(np.isnan(diag), 1.0, diag), 1e-10, 1e10)
    mean = (positions + diag**2 * gradients).mean(0)
    return mean, diag


def diag_affine(positions, gradients):
    """``(mean, scale)`` of the diagonal affine for the draws, as `make_flow`:
    from the spreads of positions and gradients, or from the gradient alone
    at a single draw."""
    positions = np.asarray(positions, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if len(positions) == 1:
        return diag_from_gradients(positions, gradients)
    pos_std = np.clip(positions.std(0), 1e-8, 1e8)
    grad_std = np.clip(gradients.std(0), 1e-8, 1e8)
    diag = np.sqrt(pos_std / grad_std)
    mean = positions.mean(0) + gradients.mean(0) * diag * diag
    return mean, diag


def _parents(sparsity, order):
    """Per position in `order`, its parents: earlier positions it shares an
    edge of the symmetrized `sparsity` with, ascending, as
    `SparseTriangularMap`."""
    if sp.issparse(sparsity):
        sparsity = sparsity.toarray()
    sparsity = np.asarray(sparsity, dtype=bool)
    dim = len(order)
    if sparsity.shape != (dim, dim):
        raise ValueError(
            f"sparsity must have shape {(dim, dim)}, got {sparsity.shape}."
        )
    blanket = sparsity[np.ix_(order, order)]
    blanket = blanket | blanket.T
    lower = np.tril(blanket, k=-1)
    return [np.flatnonzero(lower[k]).astype(np.int64) for k in range(dim)]


def default_conditioners(
    nn_width=16,
    activation="softplus",
    *,
    input_squash=None,
    scale_skip=True,
    zero_init=True,
    mlp_ridge=1.0,
    skip_ridge=0.1,
    scale_skip_ridge=1.0,
    shape_ridge=10.0,
):
    """The conditioners of a flow like `make_flow(kind="triangular")`'s: a
    constant on every coefficient, a linear conditioner on the location (and
    the scale, with `scale_skip`), and an MLP of `nn_width` units on every
    coefficient (none for ``nn_width=0``).

    With `zero_init`, a new MLP's output weights start at ``1e-3`` times
    their usual range, so the map starts close to the identity. The ridges:
    `mlp_ridge` on the MLP's weights, `skip_ridge` on the linear location
    weights, `scale_skip_ridge` on the linear scale weights, and
    `shape_ridge` on what feeds the shape coefficients directly, the
    constant and (on top of `mlp_ridge`) the MLP's output weights. What feeds
    the location is weighed against the conditional's width, see
    `FisherResiduals.ridge_strengths`."""
    conditioners = [
        {"kind": "constant", "target": "all", "ridge": {"shape": float(shape_ridge)}},
        {
            "kind": "linear",
            "target": ["location", "scale"] if scale_skip else "location",
            "ridge": {"location": float(skip_ridge), "scale": float(scale_skip_ridge)},
        },
    ]
    if nn_width:
        mlp = float(mlp_ridge)
        conditioners.append(
            {
                "kind": "mlp",
                "width": int(nn_width),
                "target": "all",
                "activation": _activation_name(activation),
                "input_squash": None if input_squash is None else float(input_squash),
                "init_output_scale": 1e-3 if zero_init else 1.0,
                "ridge": {
                    "hidden": mlp,
                    "location": mlp,
                    "scale": mlp,
                    "shape": mlp + float(shape_ridge),
                    "other": mlp,
                },
            }
        )
    return conditioners


def linear_conditioners(conditioners, transformer):
    """The linear stage of `conditioners` for the transformer kind
    `transformer` (see `transformer_kind`): the constant on its non-shape
    targets and the linear conditioner on the location, with their ridges;
    no MLPs. A fit with these moves only the Gaussian part of each
    conditional."""
    layout = transformer_layout(transformer)
    roles = [layout["location"], layout["scale"]]
    has_other = layout["n_coefs"] > len(layout["shape"]) + sum(
        role is not None for role in roles
    )
    non_shape = [
        name
        for name, present in [
            ("location", layout["location"] is not None),
            ("scale", layout["scale"] is not None),
            ("other", has_other),
        ]
        if present
    ]

    def targets(conditioner):
        target = conditioner["target"]
        return [target] if isinstance(target, str) else list(target)

    out = []
    for conditioner in conditioners:
        kind = conditioner["kind"]
        if kind == "constant":
            wanted = targets(conditioner)
            wanted = non_shape if "all" in wanted else wanted
            kept = [t for t in wanted if t != "shape"]
            if kept:
                out.append({**conditioner, "target": kept})
        elif kind == "linear":
            wanted = targets(conditioner)
            if "location" in wanted or "all" in wanted:
                out.append({**conditioner, "target": "location"})
    return out


def _seed(seed):
    """A `u64` seed for the Rust initialization from any numpy seed."""
    return int(np.random.default_rng(seed).integers(2**63))


def build_triangular_flow(
    seed,
    positions,
    gradients,
    *,
    sparsity,
    order=None,
    transformer="sas",
    transformer_layers=1,
    tangent_sas_fix_b=False,
    log_gamma_bounds=None,
    conditioners=None,
):
    """A new triangular flow for the draws.

    The diagonal affine comes from the draws and their gradients, the map's
    parents from `sparsity` in `order`. `conditioners` is a list of
    ``{"kind": ...}`` dicts, `default_conditioners()` if `None`; their
    weights start as each kind does (see `default_conditioners`), drawn from
    `seed`.
    """
    from nutpie._lib import TriangularFlow

    positions = np.asarray(positions, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)
    if positions.ndim != 2 or positions.shape != gradients.shape:
        raise ValueError("positions and gradients must both have shape (n_draw, dim).")
    if len(positions) == 0:
        raise ValueError("No draws")
    dim = positions.shape[1]
    order = np.arange(dim) if order is None else np.asarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(dim)):
        raise ValueError("order must be a permutation of range(dim).")

    kind = transformer_kind(
        transformer,
        transformer_layers,
        tangent_sas_fix_b=tangent_sas_fix_b,
        log_gamma_bounds=log_gamma_bounds,
    )
    if conditioners is None:
        conditioners = default_conditioners()
    parents = _parents(sparsity, order)
    mean, diag = diag_affine(positions, gradients)
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    return TriangularFlow(
        parent_indptr=parent_indptr,
        parent_index=(np.concatenate(parents) if dim else np.zeros(0, dtype=np.int64)),
        transformer=kind,
        conditioners=conditioners,
        permutation=order,
        loc=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(diag, dtype=np.float64),
        seed=_seed(seed),
    )


def _normalized(value):
    """`value` with tuples as lists, to compare layer dicts."""
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def _jax_transformer_kind(tmap, n_par):
    """The transformer kind of a JAX map, checked against the layers probed
    from it: the same layers, fields, indices and offsets, and the
    conditioners' skips into the kind's location and scale."""
    from nutpie.triangular_layout import (
        AsymmetricSASConfig,
        Contract2Config,
        PositiveAffineConfig,
        TangentSASConfig,
        _probe_transformer,
        transformer_dicts,
    )

    configs = _probe_transformer(tmap.transformer_constructor, n_par)
    *body, last = configs
    types = {type(config) for config in body}
    kind = None
    if not body and isinstance(last, PositiveAffineConfig):
        kind = {"kind": "affine"}
    elif not body or len(types) == 1:
        first = configs[0]
        if isinstance(last, Contract2Config) and types <= {Contract2Config}:
            kind = {"kind": "sas", "layers": len(configs)}
        elif isinstance(last, PositiveAffineConfig) and types == {TangentSASConfig}:
            kind = {
                "kind": "tangent_sas",
                "layers": len(body),
                "fix_b": first.b is None,
            }
        elif isinstance(last, PositiveAffineConfig) and types == {AsymmetricSASConfig}:
            kind = {"kind": "asymmetric_sas", "layers": len(body)}
        bounds = getattr(first, "log_gamma_bounds", None)
        if kind is not None and bounds is not None:
            kind["log_gamma_bounds"] = list(bounds)
    if kind is None:
        raise NotImplementedError("The JAX transformer is not one of the Rust kinds.")

    layout = transformer_layout(kind)
    conditioner = tmap.conditioners[0]
    scale_index = conditioner.scale_index
    if (
        layout["n_coefs"] != n_par
        or _normalized(layout["layers"]) != _normalized(transformer_dicts(configs))
        or layout["location"] != int(conditioner.location_index)
        or (scale_index is not None and layout["scale"] != int(scale_index))
    ):
        raise NotImplementedError(
            f"The JAX transformer does not match the Rust {kind['kind']!r} kind."
        )
    return kind, scale_index is not None


def from_jax(bijection):
    """The `TriangularFlow` of a JAX flow from `make_flow(kind="triangular")`,
    with the same parameters."""
    from flowjax import bijections
    from paramax import unwrap

    from nutpie._lib import TriangularFlow
    from nutpie.triangular import SparseTriangularMap
    from nutpie.triangular_lm import _parents as map_parents
    from nutpie.triangular_lm import check_supported, jax_conditioners, pack_params

    bijection = unwrap(bijection)
    inner, affine = bijection.bijections
    if isinstance(inner, bijections.Chain) and len(inner.bijections) == 1:
        inner = inner.bijections[0]
    if not (
        isinstance(inner, bijections.Sandwich)
        and isinstance(inner.inner, SparseTriangularMap)
        and isinstance(inner.outer, bijections.Permute)
        and isinstance(affine, bijections.Affine)
    ):
        raise NotImplementedError("Expected a flow from make_flow(kind='triangular').")
    tmap = inner.inner
    check_supported(tmap)
    (dim,) = tmap.shape
    parents = map_parents(tmap)
    _, second = tmap.conditioners[0].mlp.layers
    kind, _ = _jax_transformer_kind(tmap, int(second.out_features))
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    (permutation,) = inner.outer.permutation
    return TriangularFlow(
        parent_indptr=parent_indptr,
        parent_index=np.concatenate(parents).astype(np.int64),
        transformer=kind,
        conditioners=jax_conditioners(tmap),
        theta=pack_params(tmap),
        permutation=np.asarray(permutation, dtype=np.int64),
        loc=np.broadcast_to(np.asarray(affine.loc, np.float64), (dim,)).copy(),
        scale=np.broadcast_to(np.asarray(affine.scale, np.float64), (dim,)).copy(),
    )
