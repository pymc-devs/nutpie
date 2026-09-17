"""Rust backend (`src/triangular.rs`) for
`SparseTriangularMap.transform_and_log_det`, on a `TriangularLayout`.

Schedules:

``serial``
    One thread, variables in level order.
``levels``
    Each level in parallel, with a barrier between levels; levels below
    `min_parallel_work` run inline.
``dataflow``
    A variable runs as soon as its last parent is done (``rayon::join``).
``auto`` (default)
    ``serial`` if no level reaches `min_parallel_work`, else ``dataflow``.

Usage::

    from nutpie.triangular_rust import compile_transform

    fn = compile_transform(sparse_triangular_map)
    y, log_det = fn.transform_and_log_det(x)

To pull the model gradient back, record the sparse Jacobian on the way::

    y, log_det = fn.transform_and_log_det(x, record=True)
    logp, grad_y = model(y)
    grad_x = fn.pullback(grad_y)            # J^T grad_y + grad of log_det

``J = (I - A)^-1 D`` is never formed; `pullback` is one reverse sweep (see
`Tape` in `src/triangular.rs`).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from nutpie.triangular_layout import TriangularLayout, extract_layout

__all__ = [
    "compile_transform",
    "flow_transform_layout",
    "DEFAULT_MIN_PARALLEL_WORK",
    "SCHEDULES",
]


# Minimum level work (`TriangularLayout.level_work` units, ~multiply-adds)
# worth parallelizing, roughly tens of microseconds.
DEFAULT_MIN_PARALLEL_WORK = 20_000

SCHEDULES = ("auto", "serial", "levels", "dataflow")


def compile_transform(
    flow_map,
    *,
    schedule="auto",
    min_parallel_work=DEFAULT_MIN_PARALLEL_WORK,
):
    """Build the Rust forward transform for a `SparseTriangularMap` or
    `TriangularLayout`. Weights are copied in, so rebuild after refitting.

    Args:
        schedule: One of `SCHEDULES`. Use ``"serial"`` when chains already run
            in parallel.
        min_parallel_work: See `DEFAULT_MIN_PARALLEL_WORK`.
    """
    from nutpie._lib import SparseTriangularTransform

    layout = (
        flow_map if isinstance(flow_map, TriangularLayout) else extract_layout(flow_map)
    )
    return SparseTriangularTransform(
        **_transform_kwargs(layout, schedule, min_parallel_work)
    )


def flow_transform_layout(
    bijection,
    *,
    schedule="serial",
    min_parallel_work=DEFAULT_MIN_PARALLEL_WORK,
):
    """Arguments for the sampler's Rust `FlowTransform`, or `None` if the flow
    is unsupported (the sampler then uses JAX).

    Supports ``Chain([Chain([Sandwich(map, Permute)]), Affine])`` from
    ``make_flow(kind="triangular")``, and the diagonal-only
    ``Chain([Affine])``. `schedule` defaults to ``"serial"`` since chains
    already run in parallel.
    """
    from flowjax import bijections
    from paramax import unwrap

    from nutpie.triangular import SparseTriangularMap

    bijection = unwrap(bijection)
    if not isinstance(bijection, bijections.Chain):
        return None
    if len(bijection.bijections) == 1 and isinstance(
        bijection.bijections[0], bijections.Affine
    ):
        return _affine_layout(bijection.bijections[0], bijection.shape[0])
    if len(bijection.bijections) != 2:
        return None
    inner, affine = bijection.bijections
    if isinstance(inner, bijections.Chain) and len(inner.bijections) == 1:
        inner = inner.bijections[0]
    if not (
        isinstance(inner, bijections.Sandwich)
        and isinstance(inner.inner, SparseTriangularMap)
        and isinstance(inner.outer, bijections.Permute)
        and isinstance(affine, bijections.Affine)
    ):
        return None
    try:
        map_layout = extract_layout(inner.inner)
    except NotImplementedError:
        return None

    flow_layout = _transform_kwargs(map_layout, schedule, min_parallel_work)
    # flowjax keeps one index array per array axis; the flow is 1-D.
    (permutation,) = inner.outer.permutation
    flow_layout["permutation"] = np.asarray(permutation, dtype=np.int64)
    flow_layout.update(_affine_layout(affine, map_layout.n_variables))
    return flow_layout


def _affine_layout(affine, dim):
    return {
        "loc": np.broadcast_to(np.asarray(affine.loc, dtype=np.float64), (dim,)).copy(),
        "scale": np.broadcast_to(
            np.asarray(affine.scale, dtype=np.float64), (dim,)
        ).copy(),
    }


def _transform_kwargs(layout, schedule, min_parallel_work):
    """The Rust transform's constructor arguments for `layout`."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}.")

    transformer = [
        {
            field.name: list(value) if isinstance(value, tuple) else value
            for field, value in (
                (field, getattr(contract, field.name))
                for field in dataclasses.fields(contract)
            )
        }
        for contract in layout.transformer
    ]
    # Rust's `Param` is a struct, so pass (index, offset) as a map.
    for layer in transformer:
        for field in ("alpha", "beta", "sigma", "mu", "nu"):
            entry = layer[field]
            if entry is not None:
                layer[field] = {"index": int(entry[0]), "offset": float(entry[1])}

    return dict(
        parent_indptr=np.asarray(layout.parent_indptr, dtype=np.int64),
        parent_index=np.asarray(layout.parent_index, dtype=np.int64),
        blob=np.asarray(layout.blob, dtype=np.float64),
        blob_offset=np.asarray(layout.blob_offset, dtype=np.int64),
        layer_out=np.asarray(layout.layer_out, dtype=np.int64),
        skip_weight=np.asarray(layout.skip_weight, dtype=np.float64),
        skip_index=int(layout.skip_index),
        activation=layout.activation,
        transformer=transformer,
        level_ptr=np.asarray(layout.level_ptr, dtype=np.int64),
        level_vars=np.asarray(layout.level_vars, dtype=np.int64),
        level_work=np.asarray(layout.level_work, dtype=np.int64),
        min_parallel_work=int(min_parallel_work),
        schedule=schedule,
    )
