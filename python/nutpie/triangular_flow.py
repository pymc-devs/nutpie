"""Triangular flows built in Rust (`src/triangular/lm/flow.rs`), without JAX.

`build_triangular_flow` creates the flow `make_flow(kind="triangular")`
builds in JAX: the diagonal affine from the draws, the sparse triangular map
in the given order with one depth-1 MLP conditioner (with the location skip)
per variable, and its transformer. The result is a `TriangularFlow`, which
fits with `nutpie.triangular_lm.fit` on its ``residuals``, maps draws back
with ``inv_transform`` and gives the sampler its native ``flow_transform``.

Same structure and parameterization as the JAX flow, but not the same
random initialization: the random draws differ, and the map's output biases
start at exactly zero.

`from_jax` converts a JAX flow, e.g. to compare the two.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from nutpie.triangular_layout import (
    Contract2Spec,
    PositiveAffineSpec,
    TangentSASSpec,
    transformer_dicts,
)

__all__ = [
    "build_triangular_flow",
    "diag_affine",
    "diag_from_gradients",
    "from_jax",
    "transformer_spec",
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


def transformer_spec(
    *,
    contract_transformer=0,
    tangent_sas_transformer=0,
    tangent_sas_fix_b=False,
    log_gamma_bounds=None,
):
    """``(specs, n_par, location_index)`` of `make_transformer`'s chain with
    these counts (and no affine or asymmetric layers), as
    `triangular_layout._probe_transformer` finds them on the JAX transformer.

    Conditioner output ``k`` feeds one field of one layer; the fields are
    numbered in the order the JAX transformer's arrays flatten in, and every
    field starts at zero, so its offset is zero. With both counts zero, this
    is `SparseTriangularMap`'s default transformer, two `Contract2` layers.
    """
    n_contract = int(contract_transformer)
    n_tangent_sas = int(tangent_sas_transformer)
    if n_contract < 0 or n_tangent_sas < 0:
        raise ValueError("transformer counts must be non-negative.")
    if n_contract == 0 and n_tangent_sas == 0:
        n_contract = 2
    if log_gamma_bounds is not None:
        log_gamma_bounds = tuple(map(float, log_gamma_bounds))

    count = 0

    def field():
        nonlocal count
        count += 1
        return (count - 1, 0.0)

    specs = []
    location = None
    for index in range(n_contract):
        # Every interior `mu` is redundant with the next layer's `nu`, see
        # `make_transformer`; the last layer keeps it.
        alpha, beta, sigma = field(), field(), field()
        mu = field() if index == n_contract - 1 else None
        nu = field()
        specs.append(
            Contract2Spec(
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                mu=mu,
                nu=nu,
                log_gamma_bounds=log_gamma_bounds,
            )
        )
        location = mu
    for _ in range(n_tangent_sas):
        nu, eps = field(), field()
        b = None if tangent_sas_fix_b else field()
        r = field()
        specs.append(TangentSASSpec(nu=nu, eps=eps, b=b, r=r))
    if n_tangent_sas:
        loc, scale = field(), field()
        specs.append(PositiveAffineSpec(loc=loc, scale=scale))
        location = loc
    return tuple(specs), count, location[0]


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


def build_triangular_flow(
    seed,
    positions,
    gradients,
    *,
    sparsity,
    order=None,
    nn_width=16,
    activation="softplus",
    zero_init=True,
    contract_transformer=0,
    tangent_sas_transformer=0,
    tangent_sas_fix_b=False,
    log_gamma_bounds=None,
    input_squash=None,
):
    """A new triangular flow for the draws, as `make_flow(kind="triangular",
    n_layers=1, nn_depth=1, location_skip=True)` builds it.

    The diagonal affine comes from the draws and their gradients. Each
    conditioner's hidden layer is drawn uniformly in ``±1 / sqrt(fan_in)``,
    like `equinox.nn.Linear`; with `zero_init` its output layer is scaled by
    ``1e-3``, so the map starts close to the identity, as
    `normalizing_flow.zero_init_conditioners`. The map's output biases and
    the location skip start at zero.
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

    specs, n_par, location_index = transformer_spec(
        contract_transformer=contract_transformer,
        tangent_sas_transformer=tangent_sas_transformer,
        tangent_sas_fix_b=tangent_sas_fix_b,
        log_gamma_bounds=log_gamma_bounds,
    )
    parents = _parents(sparsity, order)
    n_unit = int(nn_width)

    rng = np.random.default_rng(seed)
    out_scale = 1e-3 if zero_init else 1.0
    out_limit = 1 / np.sqrt(n_unit) if n_unit else 0.0
    slices = []
    for parents_k in parents:
        n_parent = len(parents_k)
        in_limit = 1 / np.sqrt(max(n_parent, 1))
        units = []
        for _ in range(n_unit):
            units.append(rng.uniform(-in_limit, in_limit, size=n_parent + 1))
            units.append(out_scale * rng.uniform(-out_limit, out_limit, size=n_par))
        slices.extend(units)
        slices.append(np.zeros(n_par + n_parent))
    theta = np.concatenate(slices) if slices else np.zeros(0)

    mean, diag = diag_affine(positions, gradients)
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    return TriangularFlow(
        parent_indptr=parent_indptr,
        parent_index=(np.concatenate(parents) if dim else np.zeros(0, dtype=np.int64)),
        n_unit=n_unit,
        n_par=n_par,
        location_index=location_index,
        transformer=transformer_dicts(specs),
        theta=theta,
        permutation=order,
        loc=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(diag, dtype=np.float64),
        input_squash=None if input_squash is None else float(input_squash),
        activation=_activation_name(activation),
    )


def from_jax(bijection):
    """The `TriangularFlow` of a JAX flow from `make_flow(kind="triangular")`,
    with the same parameters."""
    from flowjax import bijections
    from paramax import unwrap

    from nutpie._lib import TriangularFlow
    from nutpie.triangular import SparseTriangularMap
    from nutpie.triangular_layout import _activation_name as layout_activation
    from nutpie.triangular_layout import _probe_transformer
    from nutpie.triangular_lm import _parents as map_parents
    from nutpie.triangular_lm import check_supported, pack_params

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
    first, second = tmap.conditioners[0].mlp.layers
    n_par = int(second.out_features)
    parent_indptr = np.zeros(dim + 1, dtype=np.int64)
    np.cumsum([len(p) for p in parents], out=parent_indptr[1:])
    (permutation,) = inner.outer.permutation
    return TriangularFlow(
        parent_indptr=parent_indptr,
        parent_index=np.concatenate(parents).astype(np.int64),
        n_unit=int(first.out_features),
        n_par=n_par,
        location_index=int(tmap.conditioners[0].location_index),
        transformer=transformer_dicts(
            _probe_transformer(tmap.transformer_constructor, n_par)
        ),
        theta=pack_params(tmap),
        permutation=np.asarray(permutation, dtype=np.int64),
        loc=np.broadcast_to(np.asarray(affine.loc, np.float64), (dim,)).copy(),
        scale=np.broadcast_to(np.asarray(affine.scale, np.float64), (dim,)).copy(),
        input_squash=tmap.input_squash,
        activation=layout_activation(tmap.conditioners[0].mlp.activation),
    )
