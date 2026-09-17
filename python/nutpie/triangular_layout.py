"""Flattened, backend-agnostic form of a `SparseTriangularMap`.

`SparseTriangularMap` stores its conditioners the way JAX needs them: bucketed
by parent count so that equal-width ensembles can be vmapped, padded to
rectangular levels so that a `lax.scan` can walk them. Neither is intrinsic to
the computation -- both exist to turn a ragged, sequential problem into dense
array ops.

A compiled backend (see `nutpie.triangular_numba` and
`nutpie.triangular_rust`) wants the opposite: every conditioner at its own true
input width, every parent list ragged, and the level structure kept only as
scheduling information rather than as array shape. This module performs that
translation once, so the backends only differ in how they run the resulting
loop.

Everything here is float64 and plain numpy; nothing below depends on JAX at
call time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Contract2Spec", "TriangularLayout", "extract_layout"]


_FIELDS = ("alpha", "beta", "sigma", "mu", "nu")

# Rough cost of one elementwise transformer evaluation relative to one
# multiply-add of the conditioner, used only to decide which levels are worth
# handing to a thread pool. The transformer is a handful of transcendentals per
# `Contract2` layer, which dominate the little GEMVs around them.
_TRANSFORMER_WORK_PER_LAYER = 100


@dataclass(frozen=True)
class Contract2Spec:
    """One `Contract2` layer of the elementwise transformer chain.

    Each field is either `None` (the layer does not have that parameter) or a
    ``(index, offset)`` pair meaning ``field = conditioner_output[index] +
    offset``. The offset is the layer's own initial value, which
    `get_ravelled_pytree_constructor` folds into the unravelling.
    """

    alpha: tuple[int, float] | None
    beta: tuple[int, float] | None
    sigma: tuple[int, float] | None
    mu: tuple[int, float] | None
    nu: tuple[int, float] | None
    log_gamma_bounds: tuple[float, float] | None


@dataclass(frozen=True)
class TriangularLayout:
    """A `SparseTriangularMap` flattened into ragged, per-variable arrays.

    Attributes:
        parent_indptr, parent_index: CSR parent lists. Variable ``i``'s parents
            are ``parent_index[parent_indptr[i]:parent_indptr[i + 1]]``, in the
            order the conditioner expects them, with the padded slots (which
            always read a constant zero) removed.
        blob, blob_offset: all conditioner weights, concatenated. Variable
            ``i``'s MLP occupies ``blob[blob_offset[i]:blob_offset[i + 1]]``,
            laid out layer by layer as a **transposed**, row-major
            ``(n_in, n_out)`` weight followed by an ``(n_out,)`` bias. The
            first layer's ``n_in`` is the variable's own parent count.

            The transpose is deliberate and is what makes the backends fast.
            Stored the natural way round, evaluating a layer is ``n_out`` dot
            products, each a ``+=`` reduction whose order floating point
            forbids reassociating -- so the compiler vectorizes the multiply
            and then unwinds it with shuffles to run a serial scalar add chain,
            measured at ~2.5 cycles per multiply-add. With one contiguous row
            of ``n_out`` weights per input, the inner loop is instead an AXPY
            into ``n_out`` independent accumulators, each with its own
            dependency chain, which vectorizes directly.
        layer_out: output width of each MLP layer; the last one is the number
            of transformer parameters.
        level_ptr, level_vars: elimination levels, as a second CSR. Variables
            within a level are mutually independent and may be evaluated in any
            order or in parallel; levels must be visited in order.
        level_work: estimated cost of each level, in multiply-adds.
    """

    n_variables: int
    parent_indptr: np.ndarray
    parent_index: np.ndarray
    blob: np.ndarray
    blob_offset: np.ndarray
    layer_out: np.ndarray
    activation: str
    transformer: tuple[Contract2Spec, ...]
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
    """Identify the conditioner activation.

    Matched on function identity rather than on behaviour, so an unsupported
    activation fails loudly here instead of silently sampling the wrong
    distribution.
    """
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
    """The `Contract2` layers of a transformer, as a list of field dicts."""
    from flowjax import bijections

    from nutpie.normalizing_flow import Contract2

    if isinstance(transformer, bijections.Chain):
        layers = list(transformer.bijections)
    else:
        layers = [transformer]

    out = []
    for layer in layers:
        if not isinstance(layer, Contract2):
            raise NotImplementedError(
                "The compiled transforms only support transformers built from "
                f"`Contract2` layers, got {type(layer).__name__}."
            )
        values = {}
        for field in _FIELDS:
            value = getattr(layer, field)
            values[field] = None if value is None else float(np.asarray(value))
        out.append((values, layer.log_gamma_bounds))
    return out


def _probe_transformer(constructor, num_params):
    """Recover which flat conditioner output feeds which `Contract2` field.

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
        for values, _bounds in base
    ]

    for k in range(num_params):
        probed = _chain_layers(constructor(zeros.at[k].set(1.0)))
        for layer, ((values, _), (base_values, _)) in enumerate(zip(probed, base)):
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
    for layer, (fields, (_values, bounds)) in enumerate(zip(index_map, base)):
        for field, (index, _offset) in fields.items():
            if index is None:
                raise NotImplementedError(
                    f"Transformer field {field!r} of layer {layer} is not "
                    "driven by any conditioner output."
                )
        specs.append(
            Contract2Spec(
                **{field: fields.get(field) for field in _FIELDS},
                log_gamma_bounds=None if bounds is None else tuple(map(float, bounds)),
            )
        )
    return tuple(specs)


def extract_layout(flow_map) -> TriangularLayout:
    """Flatten a `SparseTriangularMap` into a `TriangularLayout`.

    `flow_map` must have its parameters already in place -- call
    `paramax.unwrap` first if the flow still carries
    `Parameterize`/`NonTrainable` wrappers.

    Raises `NotImplementedError` if the map uses a transformer or activation
    the compiled backends do not know how to reproduce.
    """
    from nutpie.triangular import SparseTriangularMap

    if not isinstance(flow_map, SparseTriangularMap):
        raise TypeError(
            f"Expected a SparseTriangularMap, got {type(flow_map).__name__}."
        )

    (dim,) = flow_map.shape
    n_buckets = len(flow_map.conditioners)

    reference = flow_map.conditioners[0]
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

    for bucket in range(n_buckets):
        mlp = flow_map.conditioners[bucket]
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

    # Buckets exist only so that JAX can batch conditioners of equal input
    # width; here each variable gets its own ragged slice, so the bucketing is
    # flattened away and the padded parent slots (which read a constant zero,
    # and so contribute exactly nothing) are dropped with it.
    n_parents = np.zeros(dim, dtype=np.int64)
    parents_of: list[np.ndarray | None] = [None] * dim
    weights_of: list[np.ndarray | None] = [None] * dim

    for bucket in range(n_buckets):
        members = np.asarray(flow_map.bucket_members[bucket])
        parent_indices = np.asarray(flow_map.bucket_parent_indices[bucket])
        layers = [
            (
                np.asarray(layer.weight, dtype=np.float64),
                np.asarray(layer.bias, dtype=np.float64),
            )
            for layer in flow_map.conditioners[bucket].layers
        ]
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

            parts = []
            for layer_index, (weight, bias) in enumerate(layers):
                w = weight[local]
                if layer_index == 0:
                    # Trailing columns belong to padded parent slots, whose
                    # input is the constant zero, so dropping them is exact.
                    w = w[:, : len(real)]
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

    sizes = np.array([len(part) for part in weights_of], dtype=np.int64)
    blob_offset = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum(sizes, out=blob_offset[1:])
    blob = np.concatenate(weights_of) if dim else np.zeros(0)

    level_ptr, level_vars, level_work = _build_levels(
        dim, parent_indptr, parent_index, sizes, len(transformer)
    )

    buffer_size = int(max(layer_out.max(initial=1), n_parents.max(initial=0), 1))
    return TriangularLayout(
        n_variables=dim,
        parent_indptr=parent_indptr,
        parent_index=parent_index,
        blob=blob,
        blob_offset=blob_offset,
        layer_out=layer_out,
        activation=activation,
        transformer=transformer,
        level_ptr=level_ptr,
        level_vars=level_vars,
        level_work=level_work,
        buffer_size=buffer_size,
    )


def _build_levels(dim, parent_indptr, parent_index, sizes, n_transformer_layers):
    """Elimination levels, recomputed from the flattened parent lists.

    Same definition as `SparseTriangularMap` uses (``level(i)`` is the longest
    parent chain ending at ``i``), but derived here so the layout is
    self-contained and the level numbering is guaranteed to agree with the
    parent lists the backends actually read.
    """
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

    # `sizes` counts weights *and* biases, which is close enough to a
    # multiply-add count for a scheduling heuristic.
    work = sizes + _TRANSFORMER_WORK_PER_LAYER * n_transformer_layers
    level_work = np.zeros(n_levels, dtype=np.int64)
    np.add.at(level_work, level_of, work)

    return level_ptr, order.astype(np.int64), level_work
