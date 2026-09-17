"""Rust backend for `SparseTriangularMap.transform_and_log_det`.

Same flattened layout as `nutpie.triangular_numba` (see
`nutpie.triangular_layout`), but the sweep is in `src/triangular.rs` and can run
under three schedules:

``serial``
    One thread, variables in level order. No atomics, no task overhead.
``levels``
    Level-synchronous: each level's members in parallel, a barrier between
    levels, levels below `min_parallel_work` run inline.
``dataflow``
    A variable runs as soon as its last parent lands, with the fork tree built
    implicitly by ``rayon::join``. No barriers and no precomputed schedule; a
    DAG with no width never forks at all.
``auto`` (the default)
    ``serial`` when no level is wide enough to be worth parallelizing at all,
    ``dataflow`` otherwise.

Usage::

    from nutpie.triangular_rust import compile_transform

    fn = compile_transform(sparse_triangular_map)
    y, log_det = fn.transform_and_log_det(x)

For the leapfrog step the value alone is not enough -- the model's gradient has
to come back through the map -- so the forward pass can also record the sparse
Jacobian and pull a cotangent back through it::

    y, log_det = fn.transform_and_log_det(x, record=True)
    logp, grad_y = model(y)                 # between the two halves
    grad_x = fn.pullback(grad_y)            # J^T grad_y + grad of log_det

`J = dy/dx` is dense (it inverts a sparse triangular matrix), but it is never
formed: `J = (I - A)^-1 D` with `A` as sparse as the blanket, so `pullback` is
one reverse sweep over the DAG. See `Tape` in `src/triangular.rs`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from nutpie.triangular_layout import TriangularLayout, extract_layout

__all__ = ["compile_transform", "DEFAULT_MIN_PARALLEL_WORK", "SCHEDULES"]


# In the same units as `TriangularLayout.level_work` (roughly multiply-adds).
# A level has to be worth some tens of microseconds of serial work before a
# thread pool is worth reaching for at all; below that the sequential sweep
# wins outright. Used as the `levels` cutoff, and by `auto` to decide whether
# there is any width worth chasing.
DEFAULT_MIN_PARALLEL_WORK = 20_000

SCHEDULES = ("auto", "serial", "levels", "dataflow")


def compile_transform(
    flow_map,
    *,
    schedule="auto",
    min_parallel_work=DEFAULT_MIN_PARALLEL_WORK,
):
    """Build the Rust forward transform for `flow_map`.

    `flow_map` may be a `SparseTriangularMap` or an already-extracted
    `TriangularLayout`. The result bakes in the conditioner weights, so rebuild
    it whenever the flow is refitted.

    Args:
        schedule: one of `SCHEDULES`, see the module docstring. Pass
            ``"serial"`` when several chains already run in parallel, since
            they share one rayon pool.
        min_parallel_work: the `levels` cutoff, and what `auto` uses to decide
            between `serial` and `dataflow`.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}.")
    from nutpie._lib import SparseTriangularTransform

    layout = (
        flow_map if isinstance(flow_map, TriangularLayout) else extract_layout(flow_map)
    )

    transformer = [
        {
            field.name: list(value) if isinstance(value, tuple) else value
            for field, value in (
                (field, getattr(spec, field.name))
                for field in dataclasses.fields(spec)
            )
        }
        for spec in layout.transformer
    ]
    # `Param` is a struct on the Rust side, so the (index, offset) pairs have to
    # arrive as maps rather than as two-element lists.
    for layer in transformer:
        for field in ("alpha", "beta", "sigma", "mu", "nu"):
            entry = layer[field]
            if entry is not None:
                layer[field] = {"index": int(entry[0]), "offset": float(entry[1])}

    return SparseTriangularTransform(
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
