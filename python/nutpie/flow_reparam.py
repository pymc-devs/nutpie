"""Automatic flow-based reparametrization of free RVs in a PyMC model.

:func:`automatic_flow_reparam` picks flows per RV via the extensible
rewrite database :data:`flow_db`. :func:`build_flow_graph` wires the
resulting per-RV records into symbolic constrain/unconstrain maps over
Nutpie's flat point vector.
"""

from __future__ import annotations

import abc
from itertools import zip_longest

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile import optdb
from pytensor.graph.basic import Apply
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import dfs_rewriter, node_rewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery, SequenceDB
from pytensor.graph.traversal import ancestors, explicit_graph_inputs
from pytensor.tensor.math import variadic_add
from pytensor.tensor.random.basic import (
    CauchyRV,
    ExponentialRV,
    GammaRV,
    GumbelRV,
    HalfCauchyRV,
    HalfNormalRV,
    InvGammaRV,
    LaplaceRV,
    LogisticRV,
    LogNormalRV,
    MvNormalRV,
    NormalRV,
    ParetoRV,
    StudentTRV,
    WeibullRV,
)
from pytensor.xtensor.basic import XTensorFromTensor, tensor_from_xtensor


from pymc.dims.distributions.transforms import LogTransform as DimLogTransform
from pymc.distributions.transforms import log as log_transform
from pymc.logprob.transforms import LogTransform
from pymc.model.core import Model
from pymc.model.fgraph import ModelFreeRV, ModelVar, fgraph_from_model


# (loc_idx, scale_idx, required_value_transform) per RV op type. The transform
# entry is the class the ModelFreeRV's value transform must be for this RV
# op's sampling-space distribution to be loc-scale. ``None`` means no
# transform (rv and value share the same space, already loc-scale).
# ``LogTransform`` covers log-loc-scale families (LogNormal in log-space
# is Normal, so the same affine flow applies).
LOC_SCALE_FAMILIES: dict[type, tuple[int, int, type | None]] = {
    NormalRV: (0, 1, None),
    CauchyRV: (0, 1, None),
    LaplaceRV: (0, 1, None),
    LogisticRV: (0, 1, None),
    GumbelRV: (0, 1, None),
    StudentTRV: (1, 2, None),
    LogNormalRV: (0, 1, LogTransform),
}


# RV ops of the form ``X = S · Y`` (``Y`` fixed-shape, ``S`` the scale)
# that land in log-space under their default ``LogTransform``. Shifting
# the log-space value by a constant ↔ scaling ``X`` by that constant, so
# a shift-only affine flow in the sampling space captures hierarchical
# variation in the scale parameter without an icdf call. Value is the
# index of the scale/rate parameter within ``dist_params``.
SCALE_SHIFT_FAMILIES: dict[type, int] = {
    GammaRV: 1,  # β  (rate)
    WeibullRV: 1,  # β
    InvGammaRV: 1,  # β
    ExponentialRV: 0,  # λ
    HalfNormalRV: 0,  # σ
    HalfCauchyRV: 0,  # β
    ParetoRV: 1,  # m  (scale; α fixed)
}


class Flow(abc.ABC):
    """Pure-math descriptor. Subclasses declare ``n_params`` and the
    bijective ``unconstrain`` / ``constrain`` maps:

    - ``unconstrain(x)`` — value-space point → NUTS-space point (phi).
    - ``constrain(phi)`` — NUTS-space point → value-space point.

    The log|det| Jacobians default to the full Jacobian via
    ``pt.jacobian(..., vectorize=True)`` — correct for any invertible
    flow; subclasses override with a closed form when it's cheaper.
    """

    n_params: int = 0

    @staticmethod
    @abc.abstractmethod
    def unconstrain(x, *params):  # value -> phi
        ...

    @staticmethod
    @abc.abstractmethod
    def constrain(phi, *params):  # phi -> value
        ...

    @classmethod
    def log_jac_det_unconstrain(cls, x, *params):
        y = cls.unconstrain(x, *params).ravel()
        J = pt.jacobian(y, x, vectorize=True).reshape((y.size, y.size))
        return pt.linalg.slogdet(J)[1]

    @classmethod
    def log_jac_det_constrain(cls, phi, *params):
        y = cls.constrain(phi, *params).ravel()
        J = pt.jacobian(y, phi, vectorize=True).reshape((y.size, y.size))
        return pt.linalg.slogdet(J)[1]

    @classmethod
    def make_params(
        cls,
        shapes: list[tuple[int, ...]],
        dtype: str,
        name_prefix: str,
    ) -> list:
        """Create typed ``pt.tensor`` placeholders for this flow's params.
        Subclasses may override to customize names or types."""
        return [
            pt.tensor(name=f"{name_prefix}_param_{i}", dtype=dtype, shape=sh)
            for i, sh in enumerate(shapes)
        ]


class NoFlow(Flow):
    """Identity flow. Used for RVs the rewrite skipped."""

    n_params = 0

    @staticmethod
    def unconstrain(x):
        return x

    @staticmethod
    def constrain(phi):
        return phi

    @staticmethod
    def log_jac_det_unconstrain(x):
        return pt.zeros_like(x)

    @staticmethod
    def log_jac_det_constrain(phi):
        return pt.zeros_like(phi)


class AffineFlow(Flow):
    """Elementwise affine reparametrization::

        constrain(phi, shift, log_scale) = shift + exp(log_scale) * phi

    ``shift`` and ``log_scale`` broadcast against ``phi`` / ``x`` via
    normal numpy rules. The rewrite allocates them at size 1 on the axes
    where one param value suffices for the whole axis, so this class
    doesn't have to reason about shape collapse explicitly.
    """

    n_params = 2

    @staticmethod
    def unconstrain(x, shift, log_scale):
        return (x - shift) * pt.exp(-log_scale)

    @staticmethod
    def constrain(phi, shift, log_scale):
        return shift + pt.exp(log_scale) * phi

    @staticmethod
    def log_jac_det_unconstrain(x, shift, log_scale):
        return -pt.broadcast_to(log_scale, x.shape).sum()

    @staticmethod
    def log_jac_det_constrain(phi, shift, log_scale):
        return -AffineFlow.log_jac_det_unconstrain(phi, shift, log_scale)


class ShiftFlow(Flow):
    """Shift-only affine flow in the sampling space — used for
    log-transformed scale-family RVs where shifting the log-space value
    corresponds to scaling the natural-space value.

    constrain(phi, shift) = phi + shift; unconstrain is the inverse.
    Jacobian is identity (det = 1 ⇒ log|det| = 0).
    """

    n_params = 1

    @staticmethod
    def unconstrain(x, shift):
        return x - shift

    @staticmethod
    def constrain(phi, shift):
        return phi + shift

    @staticmethod
    def log_jac_det_unconstrain(x, shift):
        return pt.zeros((), dtype=x.dtype)

    @staticmethod
    def log_jac_det_constrain(phi, shift):
        return pt.zeros((), dtype=phi.dtype)


class CholeskyFlow(Flow):
    n_params = 2

    @staticmethod
    def _chol(chol_packed, n):
        """Pack a (d(d+1)/2,) vector into a (d, d) lower-triangular matrix
        with positive diagonal (diagonal entries are stored as logs)."""
        k = pt.arange(n)
        diag_indices = k * (k + 3) // 2
        chol_packed = chol_packed[..., diag_indices].set(
            pt.exp(chol_packed[..., diag_indices])
        )
        batch_shape = tuple(chol_packed.shape)[:-1]
        L = pt.zeros((*batch_shape, n, n), dtype=chol_packed.dtype)
        row_idxs, col_idxs = pt.tril_indices(n)
        L = L[..., row_idxs, col_idxs].set(chol_packed)
        return L

    @staticmethod
    def unconstrain(x, shift, chol_packed):
        n = x.shape[-1]
        L = CholeskyFlow._chol(chol_packed, n)
        return pt.linalg.solve_triangular(L, x - shift, lower=True, b_ndim=1)

    @staticmethod
    def constrain(phi, shift, chol_packed):
        n = phi.shape[-1]
        L = CholeskyFlow._chol(chol_packed, n)
        return shift + pt.matvec(L, phi)

    @staticmethod
    def log_jac_det_unconstrain(x, shift, chol_packed):
        return -CholeskyFlow.log_jac_det_constrain(x, shift, chol_packed)

    @staticmethod
    def log_jac_det_constrain(phi, shift, chol_packed):
        n = phi.shape[-1]
        k = pt.arange(n)
        diag_indices = k * (k + 3) // 2
        return chol_packed[..., diag_indices].sum(axis=-1)


class FlowFreeRV(ModelVar):
    """Marker wrapper carrying per-param shape expressions.

    Inputs (positional):
        ``(rv, value, *param_shape_exprs, *dims)``
    where ``len(param_shape_exprs) == flow_cls.n_params``; each entry is
    a 1-D int tensor giving the concrete shape of one flow parameter.
    """

    __props__ = ("flow_cls",)

    def __init__(self, flow_cls: type[Flow]):
        self.flow_cls = flow_cls
        super().__init__()

    def __call__(self, rv, value, param_shapes=(), dims=()):
        # Ergonomic construction: callers group the variadic chunks by name.
        return super().__call__(rv, value, *param_shapes, *dims)

    def make_node(self, rv, value, *rest):
        # Flat-positional so ``op.make_node(*node.inputs)`` is idempotent.
        n = self.flow_cls.n_params
        assert len(rest) >= n
        return Apply(self, [rv, value, *rest], [value.type(name=value.name)])

    def unpack(self, node):
        rv, value, *rest = node.inputs
        n = self.flow_cls.n_params
        return rv, value, rest[:n], rest[n:]


def _depends_on_free_rv(vars_) -> bool:
    return any(
        anc.owner is not None and isinstance(anc.owner.op, ModelFreeRV)
        for anc in ancestors(vars_)
    )


def _is_broadcasted(ref, *args) -> bool:
    """True iff ``ref`` has an axis that some ``arg`` unit-broadcasts
    along — i.e. the rv has event axes that loc/scale do not."""
    ref_bcast = ref.type.broadcastable
    for arg in args:
        pairs = reversed(
            tuple(zip_longest(arg.type.broadcastable, ref_bcast, fillvalue=True))
        )
        if any(a_bc and not r_bc for a_bc, r_bc in pairs):
            return True
    return False


@node_rewriter([ModelFreeRV])
def lift_xtensor_from_model_free_rv(fgraph, node):
    """Pull ``XTensorFromTensor`` wrappers out of a ``ModelFreeRV``'s rv
    and value inputs, leaving plain tensors inside so downstream flow
    rewrites don't have to know about xtensor::

        ModelFreeRV(XTensorFromTensor(rv), XTensorFromTensor(value), *dims)
        -> XTensorFromTensor(ModelFreeRV(rv, aligned_value, *dims))

    Only fires on RVs with ``transform=None`` or a :class:`DimTransform`
    known to have a plain counterpart (see ``_DIM_TRANSFORM_MAP``);
    unknown transforms are left alone since peeling might change
    semantics. Invariant: rv and value are both xtensor (or both plain
    tensors) — they may declare different dim orders, in which case the
    value is ``dimshuffle``d into rv's order so the rebuilt inner
    ``ModelFreeRV`` sees matched-axis tensors.
    """
    current_transform = node.op.transform
    # Swap any dim-aware transform for its plain logprob counterpart so
    # downstream rewrites see a single class hierarchy. Unknown transforms
    # aren't safe to peel — leave the node alone.
    match current_transform:
        case None:
            new_transform = None
        case DimLogTransform():
            new_transform = log_transform
        case _:
            return None

    xrv, xvalue, *dims = node.inputs
    if not isinstance(xrv.owner.op, XTensorFromTensor):
        return None
    rv_dims = xrv.type.dims
    value_dims = xvalue.type.dims
    if set(rv_dims) != set(value_dims):
        # Unrelated named axes — no safe permutation; leave the node alone.
        return None
    rv = xrv.owner.inputs[0]
    if xvalue.owner is not None and isinstance(xvalue.owner.op, XTensorFromTensor):
        value = xvalue.owner.inputs[0]
    else:
        value = tensor_from_xtensor(xvalue)
    value.name = xvalue.name
    if rv_dims != value_dims:
        value = value.dimshuffle([value_dims.index(d) for d in rv_dims])
        value.name = xvalue.name
    new_op = (
        node.op
        if new_transform is current_transform
        else type(node.op)(transform=new_transform)
    )
    new_free_rv = new_op(rv, value, *dims)
    return [XTensorFromTensor(dims=rv_dims)(new_free_rv)]


_one = pt.constant(1)


@node_rewriter([ModelFreeRV])
def loc_scale_affine_flow(fgraph, node):
    rv, value, *dims = node.inputs
    rv_node = rv.owner

    entry = LOC_SCALE_FAMILIES.get(type(rv_node.op))
    if entry is None:
        return None

    loc_idx, scale_idx, expected_transform = entry
    if expected_transform is None:
        if node.op.transform is not None:
            return None
    elif not isinstance(node.op.transform, expected_transform):
        return None

    dist_params = list(rv_node.op.dist_params(rv_node))
    loc, scale = dist_params[loc_idx], dist_params[scale_idx]
    if not _is_broadcasted(rv, loc, scale):
        return None
    if not _depends_on_free_rv([loc, scale]):
        return None

    rv_shape = tuple(rv.shape)
    ndim = len(rv_shape)
    batch_shape = pt.atleast_Nd(
        pt.broadcast_arrays(loc, scale)[0], n=ndim
    ).type.broadcastable
    param_shape = pt.stack(
        [_one if bc else rv_shape[i] for i, bc in enumerate(batch_shape)]
    )
    flow_rv = FlowFreeRV(AffineFlow)(
        rv, value, param_shapes=[param_shape, param_shape], dims=dims
    )
    return {node.outputs[0]: flow_rv}


@node_rewriter([ModelFreeRV])
def scale_shift_flow(fgraph, node):
    if not isinstance(node.op.transform, LogTransform):
        return None

    rv, value, *dims = node.inputs
    rv_node = rv.owner

    scale_idx = SCALE_SHIFT_FAMILIES.get(type(rv_node.op))
    if scale_idx is None:
        return None

    scale = rv_node.op.dist_params(rv_node)[scale_idx]
    if not _is_broadcasted(rv, scale):
        return None
    if not _depends_on_free_rv([scale]):
        return None

    rv_shape = tuple(rv.shape)
    ndim = len(rv_shape)
    batch_shape = pt.atleast_Nd(scale, n=ndim).type.broadcastable
    param_shape = pt.stack(
        [_one if bc else rv_shape[i] for i, bc in enumerate(batch_shape)]
    )
    flow_rv = FlowFreeRV(ShiftFlow)(rv, value, param_shapes=[param_shape], dims=dims)
    return {node.outputs[0]: flow_rv}


@node_rewriter([ModelFreeRV])
def loc_scale_cholesky_flow(fgraph, node):
    rv, value, *dims = node.inputs
    rv_node = rv.owner

    if not isinstance(rv_node.op, MvNormalRV):
        return None
    if node.op.transform is not None:
        return None

    mean, cov = rv_node.op.dist_params(rv_node)
    if not _depends_on_free_rv([mean, cov]):
        return None
    if not _is_broadcasted(rv, mean, cov[..., 0]):
        return None

    # Per-axis param size: 1 where loc & scale both broadcast against rv
    # (one shared value suffices), ``rv.shape[i]`` otherwise.
    rv_shape = tuple(rv.shape)
    ndim = len(rv_shape) - 1
    n = rv_shape[-1]
    params_bcast = pt.atleast_Nd(
        pt.broadcast_arrays(mean[..., 0], cov[..., 0, 0])[0], n=ndim
    ).type.broadcastable
    batch_shape = [_one if bc else rv_shape[i] for i, bc in enumerate(params_bcast)]
    param_shapes = [
        pt.stack([*batch_shape, n]),
        pt.stack([*batch_shape, n * (n + 1) // 2]),
    ]
    flow_rv = FlowFreeRV(CholeskyFlow)(rv, value, param_shapes=param_shapes, dims=dims)
    return {node.outputs[0]: flow_rv}


# Tag taxonomy:
#   "default" — plumbing + safe-by-default flow rewrites (used by the default
#               query); always produces correct posteriors.
#   "all"     — everything, including opt-in rewrites that a user may want
#               finer-grained control over.
# Per-flow tags ("affine", "icdf") allow targeted selection.
flow_db = SequenceDB()
flow_db.register("lower_xtensor", optdb.query("+lower_xtensor"), "default", "all")
flow_db.register(
    "lift_xtensor_from_model_free_rv",
    dfs_rewriter(lift_xtensor_from_model_free_rv),
    "default",
    "all",
)
flow_db.register(
    "affine_flow",
    dfs_rewriter(
        loc_scale_affine_flow,
        scale_shift_flow,
        loc_scale_cholesky_flow,
    ),
    "default",
    "all",
    "affine",
)


@node_rewriter([ModelVar])
def strip_model_var(fgraph, node):
    """Replace every Model*Var (including FlowFreeRV) with its first input
    so the fgraph becomes pure pytensor ops compilable via
    ``pytensor.function``."""
    return [node.inputs[0]]


strip_model_var_rewriter = dfs_rewriter(strip_model_var)


def automatic_flow_reparam(
    model: Model,
    flow_db_query: RewriteDatabaseQuery = RewriteDatabaseQuery(include=("default",)),
    db: SequenceDB = flow_db,
) -> dict[str, dict]:
    """Run the flow rewrite, strip ``ModelVar`` wrappers, evaluate the
    rewrite's per-param shape expressions in one compile, and create
    typed ``pt.tensor`` placeholders for each flow parameter.

    Returns a ``dict`` keyed by the unconstrained value variable's name
    (insertion order matches fgraph toposort). Each value is a dict with:

    ``flow_cls`` — the :class:`Flow` descriptor class (``NoFlow`` when
    the rewrite skipped the RV).
    ``dtype`` — dtype used for the flow parameters.
    ``param_shapes`` — ``list[tuple[int, ...]]`` concrete shape per param.

    Parameter placeholders are NOT created here — call
    ``flow_cls.make_params(param_shapes, dtype, name_prefix)`` when you
    actually need them (see :func:`build_flow_graph`).
    """
    fgraph, _memo = fgraph_from_model(model)
    db.query(flow_db_query).rewrite(fgraph)

    records: dict[str, dict] = {}
    all_shape_exprs: list = []
    for node in fgraph.toposort():
        op = node.op
        if isinstance(op, FlowFreeRV):
            fg_rv, fg_value, param_shape_exprs, _dims = op.unpack(node)
            records[fg_value.name] = dict(
                flow_cls=op.flow_cls,
                dtype=fg_rv.type.dtype,
            )
            all_shape_exprs.extend(param_shape_exprs)
        elif isinstance(op, ModelFreeRV):
            fg_rv, fg_value, *_dims = node.inputs
            records[fg_value.name] = dict(
                flow_cls=NoFlow,
                dtype=fg_rv.type.dtype,
            )

    strip_model_var_rewriter.rewrite(fgraph)

    if all_shape_exprs:
        inputs = list(explicit_graph_inputs(all_shape_exprs))
        shape_fn = pytensor.function(inputs, all_shape_exprs)
        ip = model.initial_point()
        all_shapes = shape_fn(**{v.name: ip[v.name] for v in inputs})
    else:
        all_shapes = []

    idx = 0
    for r in records.values():
        n = r["flow_cls"].n_params
        r["param_shapes"] = [
            tuple(int(x) for x in sh) for sh in all_shapes[idx : idx + n]
        ]
        idx += n
    return records


def build_flow_graph(
    records: dict[str, dict],
    free_vars_info,
    n_dim: int,
) -> dict[str, object]:
    """Build the symbolic constrain/unconstrain flow maps over a flat
    point vector, plus a flat flow-params vector.

    Parameters
    ----------
    records
        Output of :func:`automatic_flow_reparam`.
    free_vars_info
        Per-free-variable descriptors whose ``.name`` and ``.shape`` define
        the flat point vector layout — typically
        ``compiled_model._variables`` from ``nutpie.compile_pymc_model``
        filtered to the free/unconstrained ones. Order determines how the
        flat vector is split/packed.
    n_dim
        Total dimension of the flat point vector (``compiled_model.n_dim``).

    Returns
    -------
    dict with keys:
        ``point_vector`` — ``pt.vector`` of shape ``(n_dim,)``.
        ``flow_params_vector`` — ``pt.vector`` of shape
        ``(total_flow_params,)``.
        ``constrain`` — ``(inputs, outputs)`` pair for ``phi -> x``:
            inputs = ``[point_vector, flow_params_vector]``,
            outputs = ``[flow_point_vector, total_log_jac_det_constrain]``.
        ``unconstrain`` — ``(inputs, outputs)`` pair for ``x -> phi``.

    The caller can compile either pair with :func:`pytensor.function`
    (or hand it to a JIT backend) as they see fit.
    """
    point_vector = pt.vector("point", shape=(int(n_dim),))
    chunks = pt.unpack(point_vector, [tuple(v.shape) for v in free_vars_info])

    constrain_outs = []
    unconstrain_outs = []
    log_jac_det_constrains = []
    log_jac_det_unconstrains = []
    flow_param_inputs: list = []
    param_shapes: list[tuple[int, ...]] = []
    for v, chunk in zip(free_vars_info, chunks):
        r = records[v.name]
        flow = r["flow_cls"]
        if flow is NoFlow:
            constrain_outs.append(chunk)
            unconstrain_outs.append(chunk)
            continue
        params = flow.make_params(r["param_shapes"], r["dtype"], v.name)
        constrain_outs.append(flow.constrain(chunk, *params))
        unconstrain_outs.append(flow.unconstrain(chunk, *params))
        log_jac_det_constrains.append(flow.log_jac_det_constrain(chunk, *params).sum())
        log_jac_det_unconstrains.append(
            flow.log_jac_det_unconstrain(chunk, *params).sum()
        )
        flow_param_inputs.extend(params)
        param_shapes.extend(r["param_shapes"])

    total_ljd_constrain = variadic_add(*log_jac_det_constrains)
    total_ljd_unconstrain = variadic_add(*log_jac_det_unconstrains)

    constrain_point_vector, _ = pt.pack(*constrain_outs)
    unconstrain_point_vector, _ = pt.pack(*unconstrain_outs)
    total_flow_params = int(sum(np.prod(sh) for sh in param_shapes))
    flow_params_vector = pt.vector(
        "flow_params", dtype="float64", shape=(total_flow_params,)
    )
    if flow_param_inputs:
        flow_param_splits = pt.unpack(flow_params_vector, param_shapes)
        replace_map = dict(zip(flow_param_inputs, flow_param_splits))
        constrain_outputs = graph_replace(
            [constrain_point_vector, total_ljd_constrain], replace=replace_map
        )
        unconstrain_outputs = graph_replace(
            [unconstrain_point_vector, total_ljd_unconstrain], replace=replace_map
        )
    else:
        constrain_outputs = [constrain_point_vector, total_ljd_constrain]
        unconstrain_outputs = [unconstrain_point_vector, total_ljd_unconstrain]

    return dict(
        point_vector=point_vector,
        flow_params_vector=flow_params_vector,
        constrain=([point_vector, flow_params_vector], constrain_outputs),
        unconstrain=([point_vector, flow_params_vector], unconstrain_outputs),
    )
