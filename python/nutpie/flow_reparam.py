"""Automatic flow-based reparametrization of free RVs in a PyMC model.

:func:`reparametrize` picks flows per RV via the extensible rewrite
database :data:`flow_db` and returns plain :class:`FlowSpec` records in
ordinary rv/value/transform space (the rewrite IR never escapes it).
:func:`automatic_flow_reparam` reports the chosen flows,
:func:`build_flow_graph` turns the specs into symbolic
constrain/unconstrain maps over Nutpie's flat point vector, and
:func:`build_auto_flow` wraps those into an ``AutoFlow`` for flow
adaptation.
"""

from __future__ import annotations

import abc
import warnings
from itertools import zip_longest
from typing import NamedTuple

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile import optdb
from pytensor.graph.basic import Apply, clone_get_equiv
from pytensor.graph.fg import FunctionGraph
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
    NormalRV,
    ParetoRV,
    StudentTRV,
    WeibullRV,
)
from pytensor.xtensor.basic import (
    XTensorFromTensor,
    tensor_from_xtensor,
    xtensor_from_tensor,
)
from pytensor.xtensor.type import XTensorType


from pymc.dims.distributions.transforms import LogTransform as DimLogTransform
from pymc.distributions.multivariate import ZeroSumNormalRV
from pymc.distributions.transforms import ZeroSumTransform
from pymc.distributions.transforms import log as log_transform
from pymc.logprob.transforms import LogTransform
from pymc.logprob.utils import replace_rvs_by_values
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelFreeRV,
    ModelValuedVar,
    ModelVar,
    fgraph_from_model,
)
from pymc.pytensorf import toposort_replace


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
    """Pure-math descriptor for a per-RV reparametrization.

    Subclasses declare how many of the ``constrain`` / ``unconstrain``
    arguments come from the model graph (``n_model_params`` — e.g. the
    location/scale a child reads off its parents) versus how many are
    trainable per-flow parameters fitted during adaptation
    (``n_hyper_params`` — the VIP centering knobs). Both maps take the
    same signature ``(point, *model_params, *hyper_params)``:

    - ``unconstrain(x, ...)`` — value-space point ``x`` → NUTS-space
      point ``y``.
    - ``constrain(y, ...)`` — NUTS-space point ``y`` → value-space
      point ``x``.

    Both log|det| Jacobians default to the full Jacobian via
    ``pt.jacobian(..., vectorize=True)`` — correct for any invertible
    flow; subclasses override with a closed form when it's cheaper (see
    :class:`BaseAffineFlow` for the constant-Jacobian shortcut).
    """

    # Leading args sourced from the model fgraph (loc/scale subgraphs).
    n_model_params: int = 0
    # Trailing args fitted during adaptation; one entry per ``param_shapes``.
    n_hyper_params: int = 0

    @staticmethod
    @abc.abstractmethod
    def unconstrain(x, *params):  # x -> y
        ...

    @staticmethod
    @abc.abstractmethod
    def constrain(y, *params):  # y -> x
        ...

    @classmethod
    def log_jac_det_constrain(cls, y, *params):
        x = cls.constrain(y, *params).ravel()
        J = pt.jacobian(x, y, vectorize=True).reshape((x.size, x.size))
        return pt.linalg.slogdet(J)[1]

    @classmethod
    def log_jac_det_unconstrain(cls, x, *params):
        y = cls.unconstrain(x, *params).ravel()
        J = pt.jacobian(y, x, vectorize=True).reshape((y.size, y.size))
        return pt.linalg.slogdet(J)[1]


class BaseAffineFlow(Flow):
    """Flow that is affine in the point, i.e. its Jacobian is constant.

    For such flows the ``unconstrain`` log|det| is just the negation of
    the ``constrain`` one, and the point at which the formula is
    evaluated is irrelevant — so only ``log_jac_det_constrain`` needs a
    closed form.
    """

    @classmethod
    def log_jac_det_unconstrain(cls, x, *params):
        return -cls.log_jac_det_constrain(x, *params)


class NoFlow(BaseAffineFlow):
    """Identity flow. Used for RVs the rewrite skipped."""

    n_model_params = 0
    n_hyper_params = 0

    @staticmethod
    def unconstrain(x):
        return x

    @staticmethod
    def constrain(y):
        return y

    @staticmethod
    def log_jac_det_constrain(y):
        return pt.zeros((), dtype=y.dtype)


def _pin_if_empty(h):
    """Hyper params arrive with concrete static shapes (they are unpacked
    from the flat trainable vector); size 0 marks a knob the rewrite
    withheld because its dist param did not qualify. Pin it at the
    centred no-op ``h = 0`` so it drops out of the transform."""
    if 0 in h.type.shape:
        return pt.zeros((), dtype=h.dtype)
    return h


class AffineFlow(BaseAffineFlow):
    """Variationally Inferred Parameterisation of a location-scale RV.

    Following Gorinova et al. (2019), a child ``z ~ Dist(loc, scale)`` is
    expressed via a standardized ``y`` interpolating continuously between
    the centred (CP) and non-centred (NCP) parameterisations. A single
    trainable knob ``h`` per variable (the paper's ``1 - λ``) controls
    both location and scale, so ``h = 0`` is the no-op (centred)::

        constrain(y, loc, scale, h) = (y - (1 - h)·loc)·scale^h + loc

    ``h = 0`` ⇒ identity (CP); ``h = 1`` ⇒ full NCP (``y`` is the
    standardized residual). ``loc`` and ``scale`` are read off the parent
    RVs (model params); ``h`` is the trainable hyper param, left
    unconstrained (over-/under-centering is allowed).

    The hyper params broadcast against ``y`` / ``x`` via normal numpy
    rules. The rewrite allocates one knob per element of the value var
    (following Gorinova et al., whose λ is shaped like the RV), so
    different elements of one hierarchical group can settle on different
    centerings — except on axes where elements cannot be reparametrized
    independently (e.g. ZeroSumNormal core dims), which get size 1.

    A dist param can also fail to qualify for a knob altogether (constant
    loc, full-shape scale, ...). The rewrite then allocates its hyper
    param at size 0, and the flow pins that knob at the centred no-op
    ``h = 0`` (see :func:`_pin_if_empty`) — e.g. with ``h_sigma`` pinned
    the transform degenerates to the translation ``y + h_mu·loc``.
    """

    n_model_params = 2
    n_hyper_params = 2

    @staticmethod
    def unconstrain(x, pop_mu, pop_sigma, h_mu, h_sigma):
        h_mu, h_sigma = _pin_if_empty(h_mu), _pin_if_empty(h_sigma)
        return (x - pop_mu) * (pop_sigma**-h_sigma) + (1 - h_mu) * pop_mu

    @staticmethod
    def constrain(y, pop_mu, pop_sigma, h_mu, h_sigma):
        h_mu, h_sigma = _pin_if_empty(h_mu), _pin_if_empty(h_sigma)
        return (y - (1 - h_mu) * pop_mu) * (pop_sigma**h_sigma) + pop_mu

    @staticmethod
    def log_jac_det_constrain(y, pop_mu, pop_sigma, h_mu, h_sigma):
        # d(value)/d(y) = scale^h; h = 0 ⇒ identity.
        log_det = _pin_if_empty(h_sigma) * pt.log(pop_sigma)
        return pt.broadcast_to(log_det, y.shape).sum()


class ShiftFlow(BaseAffineFlow):
    """VIP reparametrization of a log-transformed scale-family RV.

    For ``X`` whose default ``LogTransform`` lands it in log-space, the
    parent enters the log-space value additively through ``log(scale)``,
    so non-centering is a *shift* by that amount (no scale exponent)::

        constrain(y, shift, h) = y + h·shift   with shift = log(scale)

    ``h = 0`` is the no-op (centred); ``h`` is unconstrained, so the
    optimizer reaches full decoupling regardless of whether the model
    param is a rate or a scale (the sign is absorbed into ``h``). The
    Jacobian is a translation (det = 1 ⇒ log|det| = 0).
    """

    n_model_params = 1
    n_hyper_params = 1

    @staticmethod
    def unconstrain(x, shift, h):
        return x - h * shift

    @staticmethod
    def constrain(y, shift, h):
        return y + h * shift

    @staticmethod
    def log_jac_det_constrain(y, shift, h):
        return pt.zeros((), dtype=y.dtype)


class FlowSpec(NamedTuple):
    """Per-free-RV reparametrization record in plain rv/value/transform
    space — the rewrite IR never escapes :func:`reparametrize`.

    ``rv`` is the genuine random variable (so PyMC's
    ``replace_rvs_by_values`` passes the actual distribution parameters
    to ``transform.backward``); ``model_params`` are the loc/scale
    subgraphs the flow reads off the model, expressed in terms of the
    parent specs' ``rv`` variables; ``param_shapes`` are the concrete
    shapes of the flow's trainable hyper params, evaluated at the model's
    initial point.
    """

    flow_cls: type[Flow]
    rv: object
    value: object
    transform: object | None
    model_params: list
    param_shapes: list[tuple[int, ...]]


class FlowFreeRV(ModelValuedVar):
    """Rewrite-internal marker carrying the chosen flow, the RV's value
    transform, the model-derived params, and per-hyper-param shape exprs.
    Lives only between the flow rewrite and the IR strip inside
    :func:`reparametrize`.

    Inputs (positional, flat so ``op.make_node(*node.inputs)`` is
    idempotent)::

        (rv, value, *model_params, *hyper_shape_exprs, *dims)

    with ``len(model_params) == flow_cls.n_model_params`` and
    ``len(hyper_shape_exprs) == flow_cls.n_hyper_params``.
    """

    __props__ = ("flow_cls", "transform")

    def __init__(self, flow_cls: type[Flow], transform=None):
        self.flow_cls = flow_cls
        super().__init__(transform=transform)

    def __call__(self, rv, value, *params, hyperparam_shapes=(), dims=()):
        # Ergonomic construction: callers group the variadic chunks by name.
        return super().__call__(rv, value, *params, *hyperparam_shapes, *dims)

    def make_node(self, rv, value, *rest):
        nm = self.flow_cls.n_model_params
        nh = self.flow_cls.n_hyper_params
        assert len(rest) >= nm + nh
        return Apply(self, [rv, value, *rest], [value.type(name=value.name)])


class _FlowParts(NamedTuple):
    """IR-side view of a (Flow|Model)FreeRV node, used only while the
    rewritten fgraph is alive."""

    flow_cls: type[Flow]
    rv: object
    value: object
    out: object
    transform: object | None
    model_params: list
    hyper_shapes: list


def _flow_node_parts(node) -> _FlowParts | None:
    """Extract :class:`_FlowParts` from a ``FlowFreeRV`` or plain
    ``ModelFreeRV`` node (the latter mapped to :class:`NoFlow`); returns
    ``None`` for any other node."""
    op = node.op
    if isinstance(op, FlowFreeRV):
        rv, value, *rest = node.inputs
        nm = op.flow_cls.n_model_params
        nh = op.flow_cls.n_hyper_params
        return _FlowParts(
            flow_cls=op.flow_cls,
            rv=rv,
            value=value,
            out=node.outputs[0],
            transform=op.transform,
            model_params=list(rest[:nm]),
            hyper_shapes=list(rest[nm : nm + nh]),
        )
    if isinstance(op, ModelFreeRV):
        rv, value, *_dims = node.inputs
        return _FlowParts(
            flow_cls=NoFlow,
            rv=rv,
            value=value,
            out=node.outputs[0],
            transform=op.transform,
            model_params=[],
            hyper_shapes=[],
        )
    return None


def _depends_on_free_rv(vars_) -> bool:
    return any(
        anc.owner is not None and isinstance(anc.owner.op, ModelFreeRV)
        for anc in ancestors(vars_)
    )


@node_rewriter([ModelFreeRV])
def lift_xtensor_from_model_free_rv(fgraph, node):
    """Pull ``XTensorFromTensor`` wrappers out of a ``ModelFreeRV``'s rv
    and value inputs, leaving plain tensors inside so downstream flow
    rewrites don't have to know about xtensor::

        ModelFreeRV(XTensorFromTensor(rv), XTensorFromTensor(value), *dims)
        -> XTensorFromTensor(ModelFreeRV(rv, aligned_value, *dims))

    Only fires on RVs with ``transform=None`` or a :class:`DimTransform`
    known to have a plain counterpart (see the ``match`` below); unknown
    transforms are left alone since peeling might change semantics —
    such RVs stay xtensor-typed end to end, which the spec extraction
    and graph build handle natively. Invariant: rv and value are both
    xtensor (or both plain tensors) — they may declare different dim
    orders, in which case the value is ``dimshuffle``d into rv's order
    so the rebuilt inner ``ModelFreeRV`` sees matched-axis tensors.
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
_empty_shape = pt.constant(np.array([0], dtype="int64"))


def _hyper_shape(value, n_core: int = 0):
    """Per-element hyper-param shape: the value var's shape, collapsed to
    size 1 on the trailing ``n_core`` axes (elements along support dims —
    e.g. ZeroSumNormal core dims — cannot be reparametrized independently
    and must share one knob)."""
    ndim = value.ndim
    return pt.stack(
        [*(value.shape[i] for i in range(ndim - n_core)), *([_one] * n_core)]
    )


def _qualifies_for_hyper_param(rv, param) -> bool:
    """A dist param earns a centering knob only when the RV has event
    axes the param unit-broadcasts along (a hierarchical group) and the
    param carries free-RV randomness for the knob to decouple."""
    pairs = zip_longest(
        reversed(param.type.broadcastable),
        reversed(rv.type.broadcastable),
        fillvalue=True,
    )
    if not any(p_bc and not rv_bc for p_bc, rv_bc in pairs):
        return False
    return _depends_on_free_rv([param])


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
    loc_qualifies = _qualifies_for_hyper_param(rv, loc)
    scale_qualifies = _qualifies_for_hyper_param(rv, scale)
    if not (loc_qualifies or scale_qualifies):
        return None

    # Each dist param earns its own per-element knob; a param that does
    # not qualify gets a size-0 hyper param, which the flow pins at the
    # centred no-op.
    param_shape = _hyper_shape(value)
    flow_rv = FlowFreeRV(AffineFlow, transform=node.op.transform)(
        rv,
        value,
        loc,
        scale,
        hyperparam_shapes=[
            param_shape if loc_qualifies else _empty_shape,
            param_shape if scale_qualifies else _empty_shape,
        ],
        dims=dims,
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
    if not _qualifies_for_hyper_param(rv, scale):
        return None

    # VIP shift: the parent enters the log-space value additively through
    # ``log(scale)``, so the flow shifts by ``h·log(scale)``; h = 0 is
    # centred.
    flow_rv = FlowFreeRV(ShiftFlow, transform=node.op.transform)(
        rv, value, pt.log(scale), hyperparam_shapes=[_hyper_shape(value)], dims=dims
    )
    return {node.outputs[0]: flow_rv}


@node_rewriter([ModelFreeRV])
def zerosum_scale_flow(fgraph, node):
    """VIP scale flow for ``ZeroSumNormal`` with a common (per-batch)
    sigma.

    Under the default ``ZeroSumTransform`` the value var is iid
    ``Normal(0, σ)`` over the reduced support dims (the transform is an
    isometry onto the zero-sum hyperplane and σ has core shape 1, so the
    scaling commutes with it). That makes the value-space RV a zero-loc
    scale family: :class:`AffineFlow` with ``loc = 0`` (loc knob
    withheld) applies, with one scale knob per batch element shared
    across the core dims (a per-element knob there would break the
    zero-sum coupling).
    """
    if not isinstance(node.op.transform, ZeroSumTransform):
        return None

    rv, value, *dims = node.inputs
    rv_node = rv.owner
    if not isinstance(rv_node.op, ZeroSumNormalRV):
        return None

    # ZeroSumNormalRV constructs sigma with core shape (1, ...), so it is
    # always broadcast along the core dims; guard the invariant anyway.
    sigma = rv_node.op.dist_params(rv_node)[0]
    if not all(sigma.type.broadcastable[sigma.type.ndim - rv_node.op.ndim_supp :]):
        return None
    if not _depends_on_free_rv([sigma]):
        return None

    n_core = rv_node.op.ndim_supp
    flow_rv = FlowFreeRV(AffineFlow, transform=node.op.transform)(
        rv,
        value,
        pt.zeros((), dtype=value.dtype),
        sigma,
        hyperparam_shapes=[_empty_shape, _hyper_shape(value, n_core=n_core)],
        dims=dims,
    )
    return {node.outputs[0]: flow_rv}


# Tag taxonomy:
#   "default" — plumbing + safe-by-default flow rewrites (used by the default
#               query); always produces correct posteriors.
#   "all"     — everything, including opt-in rewrites that a user may want
#               finer-grained control over.
# Per-flow tags ("affine", "icdf") allow targeted selection.
default_flow_query = RewriteDatabaseQuery(include=("default",))
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
        zerosum_scale_flow,
    ),
    "default",
    "all",
    "affine",
)


def _eval_hyper_shapes(
    model: Model, parts: list[_FlowParts]
) -> dict[str, list[tuple[int, ...]]]:
    """Evaluate every flow's symbolic hyper-param shape expressions at the
    model's initial point, in one compile. Keyed by value-var name."""
    exprs = [s for p in parts for s in p.hyper_shapes]
    if exprs:
        inputs = list(explicit_graph_inputs(exprs))
        ip = model.initial_point()
        vals = pytensor.function(inputs, exprs)(**{v.name: ip[v.name] for v in inputs})
    else:
        vals = []
    shapes: dict[str, list[tuple[int, ...]]] = {}
    idx = 0
    for p in parts:
        k = p.flow_cls.n_hyper_params
        shapes[p.value.name] = [tuple(int(x) for x in s) for s in vals[idx : idx + k]]
        idx += k
    return shapes


def _first_non_model_var(var):
    while var.owner is not None and isinstance(var.owner.op, ModelVar):
        var = var.owner.inputs[0]
    return var


def reparametrize(
    model: Model,
    flow_db_query: RewriteDatabaseQuery = default_flow_query,
    db: SequenceDB = flow_db,
) -> list[FlowSpec]:
    """Run the flow rewrite and return one :class:`FlowSpec` per free RV,
    in fgraph toposort order.

    The rewrite happens on the PyMC model IR (``fgraph_from_model``);
    afterwards every ``ModelVar`` dummy is stripped — the same in-place
    replacement ``model_from_fgraph`` performs — so the returned specs
    live in plain rv/value/transform space. PyMC's
    ``replace_rvs_by_values`` then composes parent dependencies and value
    transforms correctly on them, including conditional transforms (which
    read the RV's actual distribution parameters) and xtensor variables.
    """
    fgraph, _memo = fgraph_from_model(model)
    db.query(flow_db_query).rewrite(fgraph)
    # Per-free-RV parts, in fgraph toposort order.
    parts = [
        p for node in fgraph.toposort() if (p := _flow_node_parts(node)) is not None
    ]
    shapes = _eval_hyper_shapes(model, parts)
    # Resolve held references to non-dummy vars *before* the strip: the
    # in-place replacement below rewires the ancestors of vars that stay
    # in the fgraph, but vars removed from it keep stale inputs.
    resolved = [
        (p, [_first_non_model_var(g) for g in (p.rv, *p.model_params)]) for p in parts
    ]
    # Strip the IR in place (cf. ``model_from_fgraph``). Forward toposort
    # order: a dummy is always replaced while its consumers are still in
    # the graph, so subgraphs only reachable through a later-stripped
    # FlowFreeRV node are rewired before they are pruned.
    dummy_replacements = [
        (node.outputs[0], _first_non_model_var(node.inputs[0]))
        for node in fgraph.toposort()
        if isinstance(node.op, ModelVar)
    ]
    toposort_replace(fgraph, dummy_replacements)
    return [
        FlowSpec(
            flow_cls=p.flow_cls,
            rv=rv,
            value=p.value,
            transform=p.transform,
            model_params=model_params,
            param_shapes=shapes[p.value.name],
        )
        for p, (rv, *model_params) in resolved
    ]


def automatic_flow_reparam(
    model: Model,
    flow_db_query: RewriteDatabaseQuery = default_flow_query,
    db: SequenceDB = flow_db,
) -> dict[str, dict]:
    """Run the flow rewrite and report, per unconstrained value variable,
    which flow was chosen and the concrete shapes of its hyper params.

    Returns a ``dict`` keyed by the value variable's name (insertion order
    matches fgraph toposort). Each value is a dict with:

    ``flow_cls`` — the :class:`Flow` descriptor class (``NoFlow`` when the
    rewrite skipped the RV).
    ``dtype`` — dtype of the flow's hyper params.
    ``transform`` — the RV's value transform (``None`` or a PyMC
    :class:`Transform`).
    ``param_shapes`` — ``list[tuple[int, ...]]`` concrete shape per hyper
    param.
    """
    specs = reparametrize(model, flow_db_query, db)
    return {
        s.value.name: dict(
            flow_cls=s.flow_cls,
            dtype=s.rv.type.dtype,
            transform=s.transform,
            param_shapes=s.param_shapes,
        )
        for s in specs
    }


def build_flow_graph_from_specs(
    specs: list[FlowSpec],
    free_vars_info,
    n_dim: int,
) -> dict[str, object]:
    """Build the symbolic constrain/unconstrain flow maps over Nutpie's
    flat point vector, plus a flat trainable flow-params vector.

    Each RV's ``constrain`` / ``unconstrain`` is first expressed against
    fresh value/hyper-param placeholders with its value transform folded
    in, then PyMC's :func:`replace_rvs_by_values` composes the
    parent→child dependency in topological order. A final
    ``toposort_replace`` substitutes each value placeholder by its chunk
    of the flat point vector and each hyper placeholder by its slice of
    the flat params vector — since those are introduced last, the
    returned inputs are exact (no recovering cloned inputs by name).

    Parameters
    ----------
    specs
        Output of :func:`reparametrize`.
    free_vars_info
        Per-free-variable descriptors whose ``.name``, ``.start_idx``,
        ``.end_idx`` and ``.shape`` define the flat point vector layout —
        typically ``compiled_model._variables`` filtered to the free
        (unconstrained) ones. Order determines packing into the flat
        vector and must match Nutpie's.
    n_dim
        Total dimension of the flat point vector (``compiled_model.n_dim``).

    Returns
    -------
    dict with keys:
        ``flow_params_vector`` — flat trainable params ``pt.vector``.
        ``constrain`` — ``(inputs, outputs)`` for ``y -> value``:
            inputs ``[y_vector, flow_params_vector]``,
            outputs ``[value_point, total_log_jac_det_constrain]``.
        ``unconstrain`` — ``(inputs, outputs)`` for ``value -> y``.
    """
    n_dim = int(n_dim)
    order = [v.name for v in free_vars_info]
    info = {v.name: v for v in free_vars_info}

    # Flat trainable params vector, sliced into each flow's hyper params.
    param_shapes = [sh for s in specs for sh in s.param_shapes]
    total = int(sum(np.prod(sh) for sh in param_shapes)) if param_shapes else 0
    flow_params = pt.vector("flow_params", dtype="float64", shape=(total,))
    splits = list(pt.unpack(flow_params, param_shapes)) if param_shapes else []
    # Fresh placeholder per hyper param; substituted by its split last.
    hyper: dict[str, list] = {}
    hyper_to_split: dict = {}
    idx = 0
    for s in specs:
        phs = [
            pt.tensor(f"{s.value.name}_hyper{i}", shape=sh, dtype="float64")
            for i, sh in enumerate(s.param_shapes)
        ]
        hyper[s.value.name] = phs
        for ph in phs:
            hyper_to_split[ph] = splits[idx]
            idx += 1

    def _root_and_chunk(vec, spec):
        # The lift rewrite leaves lifted values as derived expressions
        # (``tensor_from_xtensor(xvalue)``); the flat chunk substitutes
        # the *root* value var, in its own layout.
        if spec.value.owner is None:
            root = spec.value
        else:
            (root,) = explicit_graph_inputs([spec.value])
        v = info[spec.value.name]
        chunk = vec[v.start_idx : v.end_idx].reshape(tuple(int(x) for x in v.shape))
        if isinstance(root.type, XTensorType):
            chunk = xtensor_from_tensor(chunk, dims=root.type.dims, name=root.name)
        return root, chunk

    def _build(direction: str):
        y = pt.vector("y", shape=(n_dim,))
        # Per-direction copy of the model-param subgraphs:
        # replace_rvs_by_values mutates replacement expressions in place
        # when they nest other replaced rvs (see replace_vars_in_graphs),
        # so the shared spec graphs must not be fed to it directly. The
        # rv keys are pinned to identity so they stay valid keys.
        memo = {s.rv: s.rv for s in specs}
        all_params = [p for s in specs for p in s.model_params]
        equiv = clone_get_equiv([], all_params, False, False, memo)
        model_params = {s.value.name: [equiv[p] for p in s.model_params] for s in specs}
        points: dict[str, object] = {}
        ljds: dict[str, object] = {}
        rvs_to_values: dict = {}
        replacements: list = []
        for s in specs:
            name = s.value.name
            # Fresh root placeholder for this RV's flat chunk, substituted
            # at the end; cloning inside replace_rvs_by_values keeps graph
            # inputs identical, so the substitution is exact. The value
            # var's derivation (if any) is rebuilt on top of it so the
            # flow math sees the same layout as the spec graphs.
            root, chunk = _root_and_chunk(y, s)
            z_root = root.type(name=root.name)
            replacements.append((z_root, chunk))
            if s.value is root:
                z = z_root
            else:
                memo_v = clone_get_equiv(
                    [root], [s.value], False, False, {root: z_root}
                )
                z = memo_v[s.value]
            params = model_params[name]
            if s.flow_cls is NoFlow:
                point, ljd = z, pt.zeros(())
            elif direction == "constrain":
                point = s.flow_cls.constrain(z, *params, *hyper[name])
                ljd = s.flow_cls.log_jac_det_constrain(z, *params, *hyper[name])
            else:
                point = s.flow_cls.unconstrain(z, *params, *hyper[name])
                ljd = s.flow_cls.log_jac_det_unconstrain(z, *params, *hyper[name])
            points[name], ljds[name] = point, ljd
            # A child reads this RV's *constrained* value off its parents:
            # the flow output (value space) in constrain, the value var in
            # unconstrain — backward-transformed with the RV's actual
            # distribution parameters, so conditional transforms compose
            # correctly. The backward is folded into the value here
            # instead of passing rvs_to_transforms: with transforms,
            # replace_rvs_by_values clones the graphs and remaps the
            # *keys* to clones, so rvs nested inside other replacement
            # values (a flow parent's point expression) would be missed.
            rv_value = point if direction == "constrain" else z
            if s.transform is not None:
                rv_value = s.transform.backward(rv_value, *s.rv.owner.inputs)
                rv_value = s.rv.type.filter_variable(rv_value, allow_convert=True)
                rv_value.name = s.rv.name
            rvs_to_values[s.rv] = rv_value

        graphs = [points[nm] for nm in order] + [ljds[nm] for nm in order]
        graphs = replace_rvs_by_values(graphs, rvs_to_values=rvs_to_values)
        n = len(order)
        point_parts = [
            tensor_from_xtensor(g) if isinstance(g.type, XTensorType) else g
            for g in graphs[:n]
        ]
        point_out = pt.concatenate([g.ravel() for g in point_parts])
        ljd_out = variadic_add(*graphs[n:])
        # Substitute the value/hyper placeholders by their flat-vector
        # slices. replace_rvs_by_values keeps graph inputs identical when
        # cloning, so the placeholders (and thus these replacements) are
        # exact.
        fg = FunctionGraph(outputs=[point_out, ljd_out], clone=False)
        final_replacements = [
            (ph, repl)
            for ph, repl in (*replacements, *hyper_to_split.items())
            if ph in fg.variables
        ]
        toposort_replace(fg, final_replacements)
        return [y, flow_params], list(fg.outputs)

    return dict(
        flow_params_vector=flow_params,
        constrain=_build("constrain"),
        unconstrain=_build("unconstrain"),
    )


def build_flow_graph(
    model: Model,
    free_vars_info,
    n_dim: int,
    flow_db_query: RewriteDatabaseQuery = default_flow_query,
    db: SequenceDB = flow_db,
) -> dict[str, object]:
    """:func:`reparametrize` + :func:`build_flow_graph_from_specs`."""
    specs = reparametrize(model, flow_db_query, db)
    return build_flow_graph_from_specs(specs, free_vars_info, n_dim)


def free_vars_info(compiled_model):
    """The compiled model's free (unconstrained) variable descriptors,
    whose ``start_idx``/``end_idx``/``shape`` define Nutpie's flat point
    vector layout."""
    n_dim = int(compiled_model.n_dim)
    return [v for v in compiled_model._variables if v.end_idx <= n_dim]


def build_auto_flow(
    model: Model,
    compiled_model,
    *,
    init_params=None,
    flow_db_query: RewriteDatabaseQuery = default_flow_query,
    db: SequenceDB = flow_db,
):
    """Build the VIP reparametrization of ``model`` as a single
    :class:`nutpie.normalizing_flow.AutoFlow` over ``compiled_model``'s
    flat point vector, ready to pass to
    ``compiled_model.with_transform_adapt(auto_flow=...)``.

    Prints a summary of the reparametrized variables; if the rewrite found
    nothing to reparametrize, warns and returns ``None`` instead (use
    :func:`automatic_flow_reparam` for the full per-variable report).

    The PyTensor constrain/unconstrain maps from :func:`build_flow_graph`
    are JIT-compiled to JAX. The flow's trainable parameters are the VIP
    centering knobs; ``init_params`` defaults to zeros (λ = ½, halfway
    between centred and non-centred).

    The flowjax/JAX bijection convention is ``transform: base -> target``,
    matching Nutpie's ``transform_and_log_det(sampler) -> value``; so the
    flow's ``transform`` is :func:`build_flow_graph`'s ``constrain`` and
    its ``inverse`` is ``unconstrain``.
    """
    from nutpie.normalizing_flow import AutoFlow
    import jax.numpy as jnp

    specs = reparametrize(model, flow_db_query, db)
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

    n_dim = int(compiled_model.n_dim)
    g = build_flow_graph_from_specs(specs, free_vars_info(compiled_model), n_dim)
    constrain_fn = pytensor.function(*g["constrain"], mode="JAX").vm.jit_fn
    unconstrain_fn = pytensor.function(*g["unconstrain"], mode="JAX").vm.jit_fn

    total = int(g["flow_params_vector"].type.shape[0])
    if init_params is None:
        init_params = jnp.zeros((total,))

    return AutoFlow(init_params, (n_dim,), constrain_fn, unconstrain_fn)
