"""Flattened form of a `SparseTriangularMap` for the compiled backends.

Drops the JAX layout's bucketing and padding: each conditioner gets its own
input width and a ragged parent list, and levels are only a schedule.
Plain float64 numpy.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = [
    "Contract2Spec",
    "PositiveAffineSpec",
    "TangentSASSpec",
    "TriangularLayout",
    "extract_layout",
    "transformer_dicts",
]


# Rough cost of one transformer layer, in multiply-adds, for scheduling.
_TRANSFORMER_WORK_PER_LAYER = 100


# Each spec field is `None` or ``(index, offset)``, meaning
# ``field = conditioner_output[index] + offset``. `kind` is the tag the Rust
# side dispatches on, and `FIELDS` lists the probed fields.


@dataclass(frozen=True)
class Contract2Spec:
    """One `Contract2` layer."""

    kind: ClassVar[str] = "contract2"
    FIELDS: ClassVar[tuple[str, ...]] = ("alpha", "beta", "sigma", "mu", "nu")

    alpha: tuple[int, float] | None
    beta: tuple[int, float] | None
    sigma: tuple[int, float] | None
    mu: tuple[int, float] | None
    nu: tuple[int, float] | None
    log_gamma_bounds: tuple[float, float] | None


@dataclass(frozen=True)
class TangentSASSpec:
    """One `TangentSAS` layer."""

    kind: ClassVar[str] = "tangent_sas"
    FIELDS: ClassVar[tuple[str, ...]] = ("nu", "eps", "b", "r")

    nu: tuple[int, float] | None
    eps: tuple[int, float] | None
    b: tuple[int, float] | None
    r: tuple[int, float] | None


@dataclass(frozen=True)
class PositiveAffineSpec:
    """One `PositiveAffine` layer."""

    kind: ClassVar[str] = "positive_affine"
    FIELDS: ClassVar[tuple[str, ...]] = ("loc", "scale")

    loc: tuple[int, float] | None
    scale: tuple[int, float] | None


def transformer_dicts(specs) -> list[dict]:
    """`specs` as the dicts the Rust constructors deserialize."""
    out = []
    for spec in specs:
        layer = {"kind": spec.kind}
        for field in dataclasses.fields(spec):
            value = getattr(spec, field.name)
            if field.name in spec.FIELDS:
                # Rust's `Param` is a struct, so pass (index, offset) as a map.
                value = (
                    None
                    if value is None
                    else {"index": int(value[0]), "offset": float(value[1])}
                )
            elif isinstance(value, tuple):
                value = list(value)
            layer[field.name] = value
        out.append(layer)
    return out


@dataclass(frozen=True)
class TriangularLayout:
    """A `SparseTriangularMap` flattened into ragged, per-variable arrays.

    Attributes:
        parent_indptr, parent_index: CSR parent lists, in conditioner input
            order, without padding.
        blob, blob_offset: Conditioner weights; variable ``i``'s MLP is
            ``blob[blob_offset[i]:blob_offset[i + 1]]``, per layer a
            transposed ``(n_in, n_out)`` weight then an ``(n_out,)`` bias.
            Transposed so the inner loop is a vectorizable AXPY.
        layer_out: Output width of each MLP layer.
        skip_weight, skip_index: `LocationSkipMlp` weights, aligned with
            `parent_index`, added to conditioner output `skip_index`. Zeros if
            there is no skip.
        feature_degree, feature_params: Parent features of the map. With
            degree ``K > 0`` the MLPs see ``K`` Hermite features per parent,
            parent-major, of the parent through its `marginal_to_normal` map
            with parameters ``feature_params[parent]`` (``(n_variables, 5)``).
            Degree 0: the MLPs see the raw parents.
        level_ptr, level_vars: Elimination levels as CSR. Variables within a
            level are independent.
        level_work: Estimated cost of each level, in multiply-adds.
    """

    n_variables: int
    parent_indptr: np.ndarray
    parent_index: np.ndarray
    blob: np.ndarray
    blob_offset: np.ndarray
    layer_out: np.ndarray
    skip_weight: np.ndarray
    skip_index: int
    feature_degree: int
    feature_params: np.ndarray
    activation: str
    transformer: tuple[Contract2Spec | TangentSASSpec | PositiveAffineSpec, ...]
    level_ptr: np.ndarray
    level_vars: np.ndarray
    level_work: np.ndarray
    buffer_size: int

    @property
    def n_levels(self) -> int:
        return len(self.level_ptr) - 1

    @property
    def num_params(self) -> int:
        return int(self.layer_out[-1])


def _activation_name(fn) -> str:
    """Name of the conditioner activation, matched by identity."""
    import jax
    import jax.numpy as jnp

    # jax.nn.gelu defaults to approximate=True, i.e. the tanh form.
    known = [
        (jax.nn.gelu, "gelu_tanh"),
        (jax.nn.relu, "relu"),
        (jax.nn.softplus, "softplus"),
        (jax.nn.silu, "silu"),
        (jax.nn.swish, "silu"),
        (jnp.tanh, "tanh"),
        (np.tanh, "tanh"),
    ]
    for candidate, name in known:
        if fn is candidate:
            return name
    raise NotImplementedError(
        f"Unsupported conditioner activation {fn!r}. Add it to "
        "`nutpie.triangular_layout._activation_name` and to every backend."
    )


def _is_identity(fn) -> bool:
    import jax.numpy as jnp

    probe = jnp.asarray([-1.5, 0.0, 2.25])
    try:
        return bool(np.allclose(np.asarray(fn(probe)), np.asarray(probe)))
    except Exception:  # pragma: no cover - non-elementwise final activation
        return False


def _chain_layers(transformer):
    """The layers of a transformer, as ``(spec class, field values, extra
    spec arguments)``."""
    from flowjax import bijections

    from nutpie.normalizing_flow import (
        Contract2,
        ElementwiseTransformer,
        PositiveAffine,
        TangentSAS,
    )

    if isinstance(transformer, ElementwiseTransformer):
        transformer = transformer.chain
    if isinstance(transformer, bijections.Chain):
        layers = list(transformer.bijections)
    else:
        layers = [transformer]

    out = []
    for layer in layers:
        if isinstance(layer, Contract2):
            spec, extra = Contract2Spec, {"log_gamma_bounds": layer.log_gamma_bounds}
        elif isinstance(layer, TangentSAS):
            spec, extra = TangentSASSpec, {}
        elif isinstance(layer, PositiveAffine):
            spec, extra = PositiveAffineSpec, {}
        else:
            raise NotImplementedError(
                "The compiled transforms only support transformers built from "
                "`Contract2`, `TangentSAS` and `PositiveAffine` layers, got "
                f"{type(layer).__name__}."
            )
        values = {}
        for field in spec.FIELDS:
            value = getattr(layer, field)
            values[field] = None if value is None else float(np.asarray(value))
        out.append((spec, values, extra))
    return out


def _probe_transformer(constructor, num_params):
    """Recover which flat conditioner output feeds which transformer field.

    `get_ravelled_pytree_constructor` gives a closure ``p -> unravel(p + init)``;
    the mapping from flat index to field is an implementation detail of the
    pytree flattening order, so it is measured rather than assumed. Each field
    is an affine function of exactly one parameter, so one probe per parameter
    pins the whole layout down -- and catches any structure that is not of that
    form.
    """
    import jax.numpy as jnp

    zeros = jnp.zeros((num_params,))
    base = _chain_layers(constructor(zeros))

    index_map = [
        {field: (None, value) for field, value in values.items() if value is not None}
        for _spec, values, _extra in base
    ]

    for k in range(num_params):
        probed = _chain_layers(constructor(zeros.at[k].set(1.0)))
        for layer, ((_, values, _), (_, base_values, _)) in enumerate(
            zip(probed, base)
        ):
            for field, value in values.items():
                if value is None:
                    continue
                delta = value - base_values[field]
                if abs(delta) < 1e-9:
                    continue
                if abs(delta - 1.0) > 1e-9:
                    raise NotImplementedError(
                        f"Transformer field {field!r} of layer {layer} does not "
                        "depend affinely on a single conditioner output; the "
                        "compiled transforms cannot reproduce it."
                    )
                if index_map[layer][field][0] is not None:
                    raise NotImplementedError(
                        f"Transformer field {field!r} of layer {layer} depends "
                        "on more than one conditioner output."
                    )
                index_map[layer][field] = (k, base_values[field])

    specs = []
    for layer, (fields, (spec, _values, extra)) in enumerate(zip(index_map, base)):
        for field, (index, _offset) in fields.items():
            if index is None:
                raise NotImplementedError(
                    f"Transformer field {field!r} of layer {layer} is not "
                    "driven by any conditioner output."
                )
        bounds = extra.get("log_gamma_bounds")
        if bounds is not None:
            extra = {**extra, "log_gamma_bounds": tuple(map(float, bounds))}
        specs.append(
            spec(**{field: fields.get(field) for field in spec.FIELDS}, **extra)
        )
    return tuple(specs)


def extract_layout(flow_map) -> TriangularLayout:
    """Flatten an unwrapped (`paramax.unwrap`) `SparseTriangularMap`.

    Raises `NotImplementedError` for unsupported transformers or activations.
    """
    from nutpie.triangular import LocationSkipMlp, SparseTriangularMap

    if not isinstance(flow_map, SparseTriangularMap):
        raise TypeError(
            f"Expected a SparseTriangularMap, got {type(flow_map).__name__}."
        )

    (dim,) = flow_map.shape
    n_buckets = len(flow_map.conditioners)

    skips = [
        conditioner if isinstance(conditioner, LocationSkipMlp) else None
        for conditioner in flow_map.conditioners
    ]
    mlps = [
        conditioner if skip is None else skip.mlp
        for conditioner, skip in zip(flow_map.conditioners, skips)
    ]
    skip_indices = {None if skip is None else skip.location_index for skip in skips}
    if len(skip_indices) != 1:
        raise NotImplementedError(
            "All conditioner buckets must agree on the location skip."
        )
    (skip_index,) = skip_indices
    skip_index = 0 if skip_index is None else int(skip_index)

    feature_degree = flow_map.feature_degree or 0
    features_per_parent = max(feature_degree, 1)
    feature_params = (
        np.zeros((dim, 5))
        if feature_degree == 0
        else np.asarray(flow_map.feature_params, dtype=np.float64)
    )

    reference = mlps[0]
    n_layers = len(reference.layers)
    activation = _activation_name(reference.activation)
    if not _is_identity(reference.final_activation):
        raise NotImplementedError(
            "The compiled transforms assume the conditioner's final activation "
            "is the identity."
        )
    layer_out = np.array(
        [int(layer.out_features) for layer in reference.layers], dtype=np.int64
    )
    num_params = int(layer_out[-1])

    for mlp in mlps:
        if len(mlp.layers) != n_layers or mlp.activation is not reference.activation:
            raise NotImplementedError(
                "All conditioner buckets must share the same depth and activation."
            )
        for layer, ref_layer in zip(mlp.layers, reference.layers):
            if layer.bias is None:
                raise NotImplementedError(
                    "The compiled transforms require conditioners with biases."
                )
            if layer.out_features != ref_layer.out_features:
                raise NotImplementedError(
                    "All conditioner buckets must share the same layer widths."
                )

    transformer = _probe_transformer(flow_map.transformer_constructor, num_params)

    # Unbucket, dropping padded parent slots (they read a constant zero).
    n_parents = np.zeros(dim, dtype=np.int64)
    parents_of: list[np.ndarray | None] = [None] * dim
    weights_of: list[np.ndarray | None] = [None] * dim
    skip_of: list[np.ndarray | None] = [None] * dim

    for bucket in range(n_buckets):
        members = np.asarray(flow_map.bucket_members[bucket])
        parent_indices = np.asarray(flow_map.bucket_parent_indices[bucket])
        layers = [
            (
                np.asarray(layer.weight, dtype=np.float64),
                np.asarray(layer.bias, dtype=np.float64),
            )
            for layer in mlps[bucket].layers
        ]
        skip = (
            None
            if skips[bucket] is None
            else np.asarray(skips[bucket].skip, dtype=np.float64)
        )
        for local, variable in enumerate(members):
            variable = int(variable)
            row = parent_indices[local]
            real = row[row != dim].astype(np.int64)
            if np.any(real >= variable):
                raise ValueError(
                    f"Variable {variable} has a parent that does not precede it."
                )
            n_parents[variable] = len(real)
            parents_of[variable] = real
            # Real parents fill the leading slots, as for the first layer.
            skip_of[variable] = (
                np.zeros(len(real)) if skip is None else skip[local, : len(real)]
            )

            parts = []
            for layer_index, (weight, bias) in enumerate(layers):
                w = weight[local]
                if layer_index == 0:
                    w = w[:, : len(real) * features_per_parent]
                # Transposed to (n_in, n_out); see `TriangularLayout.blob`.
                parts.append(np.ascontiguousarray(w.T).ravel())
                parts.append(bias[local])
            weights_of[variable] = np.concatenate(parts)

    missing = [i for i, part in enumerate(weights_of) if part is None]
    if missing:
        raise ValueError(f"Variables {missing} are not covered by any bucket.")

    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum(n_parents, out=parent_indptr[1:])
    parent_index = (
        np.concatenate(parents_of).astype(np.int64)
        if dim
        else np.zeros(0, dtype=np.int64)
    )
    skip_weight = np.concatenate(skip_of) if dim else np.zeros(0)

    sizes = np.array([len(part) for part in weights_of], dtype=np.int64)
    blob_offset = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum(sizes, out=blob_offset[1:])
    blob = np.concatenate(weights_of) if dim else np.zeros(0)

    level_ptr, level_vars, level_work = _build_levels(
        dim, parent_indptr, parent_index, sizes, len(transformer)
    )

    buffer_size = int(
        max(
            layer_out.max(initial=1),
            n_parents.max(initial=0) * features_per_parent,
            1,
        )
    )
    return TriangularLayout(
        n_variables=dim,
        parent_indptr=parent_indptr,
        parent_index=parent_index,
        blob=blob,
        blob_offset=blob_offset,
        layer_out=layer_out,
        skip_weight=skip_weight,
        skip_index=skip_index,
        feature_degree=int(feature_degree),
        feature_params=feature_params,
        activation=activation,
        transformer=transformer,
        level_ptr=level_ptr,
        level_vars=level_vars,
        level_work=level_work,
        buffer_size=buffer_size,
    )


def _build_levels(dim, parent_indptr, parent_index, sizes, n_transformer_layers):
    """Elimination levels (longest parent chain), recomputed from the
    flattened parent lists."""
    level_of = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        parents = parent_index[parent_indptr[i] : parent_indptr[i + 1]]
        if len(parents):
            level_of[i] = level_of[parents].max() + 1

    n_levels = int(level_of.max(initial=-1)) + 1
    order = np.argsort(level_of, kind="stable")
    counts = np.bincount(level_of, minlength=n_levels)
    level_ptr = np.zeros(n_levels + 1, dtype=np.int64)
    np.cumsum(counts, out=level_ptr[1:])

    work = sizes + _TRANSFORMER_WORK_PER_LAYER * n_transformer_layers
    level_work = np.zeros(n_levels, dtype=np.int64)
    np.add.at(level_work, level_of, work)

    return level_ptr, order.astype(np.int64), level_work
