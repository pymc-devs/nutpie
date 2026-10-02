from collections.abc import Callable

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from flowjax import bijections
from flowjax.utils import get_ravelled_pytree_constructor
from jax import Array
from jax.typing import ArrayLike
from paramax import NonTrainable


def _min_waste_segments(
    counts: np.ndarray, n_segments: int, weights: np.ndarray | None = None
) -> list:
    """Split `counts` into at most `n_segments` contiguous runs, minimizing
    padding ``sum(len(run) * run_max)``; returns ``(start, stop)`` pairs.

    Like `_min_waste_buckets`, but the order is fixed: levels cannot be
    reordered. For ``(n, n_groups)`` counts each group is padded separately,
    and the cost is ``len(run) * sum_g weights[g] * run_max[g]``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim == 1:
        counts = counts[:, None]
    n, n_groups = counts.shape
    n_segments = max(1, min(n_segments, n))
    if weights is None:
        weights = np.ones(n_groups, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    no_split = -1
    prev_dp = np.full(n + 1, np.inf)
    prev_dp[0] = 0.0
    split = np.full((n_segments + 1, n + 1), no_split, dtype=np.int64)
    for b in range(1, n_segments + 1):
        new_dp = np.full(n + 1, np.inf)
        for r in range(b, n + 1):
            l_range = np.arange(b - 1, r)
            # running max of counts[l:r], for every candidate left edge l
            run_max = np.maximum.accumulate(counts[l_range][::-1], axis=0)[::-1]
            costs = prev_dp[l_range] + (run_max @ weights) * (r - l_range)
            best = int(np.argmin(costs))
            new_dp[r] = costs[best]
            split[b, r] = l_range[best]
        prev_dp = new_dp

    boundaries = []
    r = n
    for b in range(n_segments, 0, -1):
        left = int(split[b, r])
        if left == no_split:
            break
        boundaries.append((left, r))
        r = left
        if r == 0:
            break
    boundaries.reverse()
    return boundaries


def _min_waste_buckets(counts: np.ndarray, n_buckets: int) -> np.ndarray:
    """Partition `counts` into at most `n_buckets` groups, minimizing padding
    ``sum(group_max - value)``; returns each item's bucket (0 = smallest).

    Optimal groups are contiguous in sorted order, so this is an exact
    ``O(n^2 * n_buckets)`` dynamic program.
    """
    counts = np.asarray(counts)
    n = len(counts)
    n_buckets = max(1, min(n_buckets, n))
    order = np.argsort(counts, kind="stable")
    sorted_counts = counts[order].astype(np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(sorted_counts)])

    # dp[r] = min total waste covering the first r (sorted) items with the
    # number of buckets processed so far; split[b, r] = the best boundary.
    no_split = -1
    prev_dp = np.full(n + 1, np.inf)
    prev_dp[0] = 0.0
    split = np.full((n_buckets + 1, n + 1), no_split, dtype=np.int64)
    for b in range(1, n_buckets + 1):
        new_dp = np.full(n + 1, np.inf)
        for r in range(b, n + 1):
            l_range = np.arange(b - 1, r)
            # cost of segment [l, r): padding sorted_counts[l:r] up to
            # sorted_counts[r - 1] (the segment's max, since sorted
            # ascending).
            costs = prev_dp[l_range] + (
                sorted_counts[r - 1] * (r - l_range) - (prefix[r] - prefix[l_range])
            )
            best = int(np.argmin(costs))
            new_dp[r] = costs[best]
            split[b, r] = l_range[best]
        prev_dp = new_dp

    boundaries = []
    r = n
    for b in range(n_buckets, 0, -1):
        left = int(split[b, r])
        boundaries.append((left, r))
        r = left
    boundaries.reverse()

    bucket_of_sorted = np.zeros(n, dtype=np.int64)
    for bucket_idx, (left, right) in enumerate(boundaries):
        bucket_of_sorted[left:right] = bucket_idx

    bucket_of_item = np.zeros(n, dtype=np.int64)
    bucket_of_item[order] = bucket_of_sorted
    return bucket_of_item


class _SparseTriangularLayout(eqx.Module):
    """Static sparsity structure of the Jacobian, built once outside jit.

    ``level_*`` groups edges by their parent's level, for back substitution
    (``J^T w = rhs``). ``child_level_*`` groups them by their child's level,
    for forward substitution, which differentiating that solve needs.
    """

    # Per bucket: flat positions within bucket_jacobian_rows[bucket].ravel()
    # that correspond to real (non-padding) edges.
    bucket_value_gather: tuple
    # (n_edges + 1,) child variable of each edge; final entry is the
    # sentinel, which reads the constant zero appended to the solution
    # during the solve.
    edge_child_index: jnp.ndarray
    # (n_edges + 1,) parent variable of each edge, same sentinel convention.
    edge_parent_index: jnp.ndarray
    # Tuples with one array per contiguous segment of levels, each at its own
    # width (see `_min_waste_segments`), in level order.
    #
    # level_members:           (levels_in_segment, max_level_size), padded with `dim`
    # *_edge_index:            (levels_in_segment, width) edges landing on that level
    # *_edge_target_slot:      (levels_in_segment, width) slot within level_members;
    #                          padding uses max_level_size, out of bounds and dropped
    level_members: tuple
    # Grouped by the level of each edge's *parent* (back substitution).
    level_edge_index: tuple
    level_edge_target_slot: tuple
    # Grouped by the level of each edge's *child* (the transposed solve).
    child_level_edge_index: tuple
    child_level_edge_target_slot: tuple

    n_variables: int = eqx.field(static=True)
    n_edges: int = eqx.field(static=True)
    max_level_size: int = eqx.field(static=True)


def _build_layout(
    bucket_members,
    bucket_parent_indices,
    level_of_variable,
    dim,
    n_level_segments,
):
    """Build the `_SparseTriangularLayout` from the bucket and level data."""
    edge_child, edge_parent, bucket_value_gather = [], [], []

    for bucket in range(len(bucket_members)):
        parent_indices = np.asarray(bucket_parent_indices[bucket])
        members = np.asarray(bucket_members[bucket])
        is_real_edge = parent_indices.ravel() != dim  # sentinel padding
        real_positions = np.flatnonzero(is_real_edge)

        bucket_value_gather.append(real_positions)
        n_parent_slots = parent_indices.shape[1]
        edge_child.append(np.repeat(members, n_parent_slots)[real_positions])
        edge_parent.append(parent_indices.ravel()[real_positions])

    edge_child = np.concatenate(edge_child)
    edge_parent = np.concatenate(edge_parent)
    n_edges = len(edge_child)

    level_of_variable = np.asarray(level_of_variable)
    n_levels = int(level_of_variable.max()) + 1
    level_sizes = np.bincount(level_of_variable, minlength=n_levels)
    max_level_size = int(level_sizes.max())

    level_members = np.full((n_levels, max_level_size), dim, np.int32)
    slot_within_level = np.zeros(dim, np.int32)
    for level in range(n_levels):
        members_at_level = np.flatnonzero(level_of_variable == level)
        level_members[level, : len(members_at_level)] = members_at_level
        slot_within_level[members_at_level] = np.arange(len(members_at_level))

    def group_edges(level_of_edge, endpoint_of_edge, segments):
        """Per-segment (edge index, target slot) arrays."""
        by_level = [np.flatnonzero(level_of_edge == level) for level in range(n_levels)]
        index_segments, slot_segments = [], []
        for start, stop in segments:
            width = max((len(by_level[l]) for l in range(start, stop)), default=0)
            width = max(width, 1)
            index = np.full((stop - start, width), n_edges, np.int32)
            slot = np.full((stop - start, width), max_level_size, np.int32)
            for row, level in enumerate(range(start, stop)):
                edges = by_level[level]
                index[row, : len(edges)] = edges
                slot[row, : len(edges)] = slot_within_level[endpoint_of_edge[edges]]
            index_segments.append(jnp.asarray(index))
            slot_segments.append(jnp.asarray(slot))
        return tuple(index_segments), tuple(slot_segments)

    # Each grouping gets its own segmentation; their per-level counts differ.
    level_of_edge_parent = level_of_variable[edge_parent]
    level_of_edge_child = level_of_variable[edge_child]
    parent_counts = np.bincount(level_of_edge_parent, minlength=n_levels)
    child_counts = np.bincount(level_of_edge_child, minlength=n_levels)
    parent_segments = _min_waste_segments(parent_counts, n_level_segments)
    child_segments = _min_waste_segments(child_counts, n_level_segments)

    level_edge_index, level_edge_target_slot = group_edges(
        level_of_edge_parent, edge_parent, parent_segments
    )
    child_level_edge_index, child_level_edge_target_slot = group_edges(
        level_of_edge_child, edge_child, child_segments
    )
    # `level_members` is sliced to match each grouping's segmentation.
    parent_members = tuple(
        jnp.asarray(level_members[start:stop]) for start, stop in parent_segments
    )
    child_members = tuple(
        jnp.asarray(level_members[start:stop]) for start, stop in child_segments
    )

    return _SparseTriangularLayout(
        bucket_value_gather=tuple(jnp.asarray(g) for g in bucket_value_gather),
        edge_child_index=jnp.asarray(np.append(edge_child, dim).astype(np.int32)),
        edge_parent_index=jnp.asarray(np.append(edge_parent, dim).astype(np.int32)),
        level_members=(parent_members, child_members),
        level_edge_index=level_edge_index,
        level_edge_target_slot=level_edge_target_slot,
        child_level_edge_index=child_level_edge_index,
        child_level_edge_target_slot=child_level_edge_target_slot,
        n_variables=dim,
        n_edges=n_edges,
        max_level_size=max_level_size,
    )


@jax.profiler.annotate_function
def _flatten_edge_values(bucket_jacobian_rows, layout):
    """Per-bucket Jacobian rows -> flat ``(n_edges + 1,)`` edge values."""
    per_bucket = [
        rows.ravel()[gather]
        for rows, gather in zip(bucket_jacobian_rows, layout.bucket_value_gather)
    ]
    sentinel = jnp.zeros((1,), per_bucket[0].dtype)
    return jnp.concatenate(per_bucket + [sentinel])


@jax.profiler.annotate_function
def _sweep_levels(
    edge_values,
    jacobian_diagonal,
    layout,
    rhs,
    edge_endpoint_index,
    level_members,
    level_edge_index,
    level_edge_target_slot,
    reverse,
):
    """One triangular solve, as one scan per level segment.

    `edge_endpoint_index` picks the already-solved endpoint of each edge: the
    child for back substitution, the parent for forward substitution.
    """
    max_level_size = layout.max_level_size

    # Sentinel row: rhs 0 and diagonal 1, so padded lanes cannot produce NaNs
    # that would reach a cotangent.
    rhs_padded = jnp.concatenate([rhs, jnp.zeros((1,), rhs.dtype)])
    diagonal_padded = jnp.concatenate([jacobian_diagonal, jnp.ones((1,), rhs.dtype)])

    @jax.profiler.annotate_function
    def eliminate_level(solution, level_data):
        members, edge_indices, target_slots = level_data
        solution_padded = jnp.concatenate([solution, jnp.zeros((1,), rhs.dtype)])
        solved_values = solution_padded[edge_endpoint_index[edge_indices]]
        edge_contributions = edge_values[edge_indices] * solved_values

        # Padding edges target the out-of-bounds slot `max_level_size`.
        neighbour_sum = (
            jnp.zeros((max_level_size,), rhs.dtype)
            .at[target_slots]
            .add(edge_contributions, mode="drop")
        )

        updated = (rhs_padded[members] - neighbour_sum) / diagonal_padded[members]
        solution = solution.at[members].set(updated, mode="drop", unique_indices=True)
        return solution, None

    segments = list(zip(level_members, level_edge_index, level_edge_target_slot))
    if reverse:
        segments = segments[::-1]

    solution = jnp.zeros_like(rhs)
    for members, edge_index, target_slot in segments:
        solution, _ = jax.lax.scan(
            eliminate_level,
            solution,
            (members, edge_index, target_slot),
            reverse=reverse,
        )
    return solution


def _solve_triangular_sparse(edge_values, jacobian_diagonal, layout, rhs):
    """Solve ``J^T w = rhs`` by back substitution.

    Uses `jax.lax.custom_linear_solve`: autodiff through the scan would store
    one ``(dim,)`` carry per level, and ``n_levels == dim`` is common.
    """

    def matvec(w):
        """``J^T w`` as one scatter over the edges."""
        w_padded = jnp.concatenate([w, jnp.zeros((1,), w.dtype)])
        contributions = edge_values * w_padded[layout.edge_child_index]
        return jacobian_diagonal * w + jnp.zeros_like(w).at[
            layout.edge_parent_index
        ].add(contributions, mode="drop")

    def back_substitute(_, b):
        return _sweep_levels(
            edge_values,
            jacobian_diagonal,
            layout,
            b,
            layout.edge_child_index,
            layout.level_members[0],
            layout.level_edge_index,
            layout.level_edge_target_slot,
            reverse=True,
        )

    def forward_substitute(_, b):
        return _sweep_levels(
            edge_values,
            jacobian_diagonal,
            layout,
            b,
            layout.edge_parent_index,
            layout.level_members[1],
            layout.child_level_edge_index,
            layout.child_level_edge_target_slot,
            reverse=False,
        )

    return jax.lax.custom_linear_solve(
        matvec, rhs, back_substitute, transpose_solve=forward_substitute
    )


class _SelectedInverseLayout(eqx.Module):
    """Static index structure for `_selected_inverse`, built once outside jit.

    ``Sigma = (J^T J)^{-1}`` is needed only on each ``{i} + P(i)``. From
    ``J Sigma = J^{-T}`` (Takahashi's recurrence):

        Sigma[i, j] = ([i == j] / delta_i - sum_{p in P(i)} A[i, p] Sigma[p, j]) / delta_i

    This closes only if each ``P(i)`` is a clique; ``P*(i)`` adds the fill
    that makes it one.

    Store layout: the diagonal at ``[0, dim)``, one slot per
    ``(i, j in P*(i))``, a zero `sentinel` for padded reads, and a `dump`
    slot for padded writes.

    level_rows:   (levels, R)          row indices, padded with `dim`
    a_edge:       (levels, R, K)       edge index of `A[i, p]`, padded with `n_edges`
    lookup:       (levels, R, K, K*)   store index of `Sigma[p, j]`, padded with `sentinel`
    out:          (levels, R, K*)      store index of `Sigma[i, j]`, padded with `dump`
    diag_from:    (levels, R, K)       position of `p` within `P*(i)`, padded with `K*`
    """

    level_rows: Array
    a_edge: Array
    lookup: Array
    out: Array
    diag_from: Array
    store_size: int = eqx.field(static=True)


def _build_selected_inverse(edge_child, edge_parent, dim):
    """`_SelectedInverseLayout` plus a `store_index(a, b)` for `Sigma[a, b]`."""
    edge_child = np.asarray(edge_child)[:-1]  # drop the sentinel edge
    edge_parent = np.asarray(edge_parent)[:-1]
    n_edges = len(edge_child)

    parents = [[] for _ in range(dim)]
    edge_of = {}
    for edge, (child, parent) in enumerate(zip(edge_child, edge_parent)):
        parents[int(child)].append(int(parent))
        edge_of[(int(child), int(parent))] = edge

    # Symbolic fill: each `p` in `P*(i)` must see the earlier part of `P*(i)`.
    # Walking backwards finalizes `P*(i)` before it propagates.
    filled = [set(ps) for ps in parents]
    for i in reversed(range(dim)):
        members = sorted(filled[i])
        for p in members:
            filled[p].update(q for q in members if q < p)
    filled = [sorted(s) for s in filled]

    slot = {}
    for i in range(dim):
        for j in filled[i]:
            slot[(i, j)] = dim + len(slot)
    sentinel = dim + len(slot)
    dump = sentinel + 1

    def store_index(a, b):
        if a == b:
            return a
        return slot[(max(a, b), min(a, b))]

    level = np.zeros(dim, dtype=np.int64)
    for i in range(dim):
        if filled[i]:
            level[i] = 1 + max(level[j] for j in filled[i])
    n_levels = int(level.max(initial=0)) + 1
    by_level = [np.flatnonzero(level == lvl) for lvl in range(n_levels)]
    R = max(len(rows) for rows in by_level)
    K = max(1, max((len(ps) for ps in parents), default=0))
    K_star = max(1, max((len(s) for s in filled), default=0))

    level_rows = np.full((n_levels, R), dim, np.int32)
    a_edge = np.full((n_levels, R, K), n_edges, np.int32)
    lookup = np.full((n_levels, R, K, K_star), sentinel, np.int32)
    out = np.full((n_levels, R, K_star), dump, np.int32)
    diag_from = np.full((n_levels, R, K), K_star, np.int32)
    for lvl, rows in enumerate(by_level):
        for r, i in enumerate(rows):
            level_rows[lvl, r] = i
            for m, j in enumerate(filled[i]):
                out[lvl, r, m] = slot[(i, j)]
            for k, p in enumerate(parents[i]):
                a_edge[lvl, r, k] = edge_of[(i, p)]
                diag_from[lvl, r, k] = filled[i].index(p)
                for m, j in enumerate(filled[i]):
                    lookup[lvl, r, k, m] = store_index(p, j)

    layout = _SelectedInverseLayout(
        level_rows=jnp.asarray(level_rows),
        a_edge=jnp.asarray(a_edge),
        lookup=jnp.asarray(lookup),
        out=jnp.asarray(out),
        diag_from=jnp.asarray(diag_from),
        store_size=dump + 1,
    )
    return layout, store_index, sentinel


def _selected_inverse(edge_values, jacobian_diagonal, layout):
    """``(J^T J)^{-1}`` on the filled pattern, as `_SelectedInverseLayout`'s
    flat store."""
    diagonal = jnp.concatenate([jacobian_diagonal, jnp.ones((1,), edge_values.dtype)])
    store = jnp.zeros((layout.store_size,), edge_values.dtype)

    def row_level(store, level_data):
        rows, a_edge, lookup, out, diag_from = level_data
        delta = diagonal[rows]  # (R,)
        a = edge_values[a_edge]  # (R, K)
        # Sigma[i, j] for j in P*(i): all read entries sit at earlier levels.
        off = -jnp.einsum("rk,rkm->rm", a, store[lookup]) / delta[:, None]
        store = store.at[out].set(off)
        # Sigma[i, i] needs Sigma[i, p] for p in P(i), just computed.
        off_padded = jnp.concatenate([off, jnp.zeros((off.shape[0], 1), off.dtype)], 1)
        sigma_ip = jnp.take_along_axis(off_padded, diag_from, axis=1)
        diag = (1.0 / delta - jnp.sum(a * sigma_ip, axis=1)) / delta
        # Padded rows (`dim`) go to the scratch slot.
        diag_out = jnp.where(
            rows < jacobian_diagonal.shape[0], rows, layout.store_size - 1
        )
        store = store.at[diag_out].set(diag)
        return store, None

    store, _ = jax.lax.scan(
        row_level,
        store,
        (layout.level_rows, layout.a_edge, layout.lookup, layout.out, layout.diag_from),
    )
    return store


def marginal_to_normal(params: Array, y: Array) -> tuple[Array, Array]:
    """``u = g(y)`` and ``log g'(y)`` for the monotone marginal map of one
    coordinate, a sinh-arcsinh (`Contract2`) inverse.

    ``params[..., :]`` are ``(log gamma, eps, log sigma, mu, nu)``; zeros give
    the identity. `fit_marginal_maps` fits them so that ``u`` is standard
    normal under the draws.
    """
    log_gamma, eps, log_sigma, mu, nu = jnp.moveaxis(params, -1, 0)
    half = jnp.exp(log_gamma - log_sigma) * (y - mu) / 2
    w = (jnp.arcsinh(half) - eps) * jnp.exp(-log_gamma)
    u = 2 * jnp.sinh(w) + nu
    log_du = -log_sigma + _log_cosh(w) - 0.5 * jnp.log1p(half * half)
    return u, log_du


def _log_cosh(v):
    a = jnp.abs(v)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - jnp.log(2.0)


def hermite_features(u: Array, degree: int) -> Array:
    """``He_k(u) / sqrt(k!)`` for ``k = 1..degree``, stacked on a new last
    axis: orthonormal under a standard normal ``u``."""
    previous, current = jnp.ones_like(u), u
    out = [current]
    for k in range(1, degree):
        previous, current = current, u * current - k * previous
        out.append(current / np.sqrt(float(np.prod(np.arange(1, k + 2)))))
    return jnp.stack(out, axis=-1)


def fit_marginal_maps(y, *, steps: int = 100, ridge: float = 1e-3) -> np.ndarray:
    """Fit `marginal_to_normal` per coordinate of the draws ``y`` (``(n, dim)``)
    by maximum likelihood, so that ``u`` is standard normal. Returns
    ``(dim, 5)`` parameters.

    Damped Newton, vectorized over the coordinates. The small `ridge`
    towards the identity pins down ``mu`` and ``nu``, which are redundant
    for a near normal marginal.
    """
    y = jnp.asarray(y)

    def loss(theta, values):
        u, log_du = marginal_to_normal(theta, values)
        return jnp.mean(0.5 * u * u - log_du) + ridge * jnp.sum(theta * theta)

    loss_all = jax.vmap(loss, in_axes=(0, 1))
    grad_all = jax.vmap(jax.grad(loss), in_axes=(0, 1))
    hess_all = jax.vmap(jax.hessian(loss), in_axes=(0, 1))

    def step(_, state):
        theta, value, lam = state
        g, H = grad_all(theta, y), hess_all(theta, y)
        damped = H + lam[:, None, None] * jnp.eye(5)
        proposal = theta + jnp.linalg.solve(damped, -g[..., None])[..., 0]
        new_value = loss_all(proposal, y)
        better = jnp.isfinite(new_value) & (new_value < value)
        theta = jnp.where(better[:, None], proposal, theta)
        value = jnp.where(better, new_value, value)
        lam = jnp.where(better, lam / 3, lam * 4)
        return theta, value, lam

    dim = y.shape[1]
    theta = jnp.zeros((dim, 5), y.dtype)
    state = (theta, loss_all(theta, y), jnp.full(dim, 1e-2, y.dtype))
    theta, _, _ = jax.jit(lambda s: jax.lax.fori_loop(0, steps, step, s))(state)
    return np.asarray(theta)


class LocationSkipMlp(eqx.Module):
    """Conditioner MLP plus a linear skip from the parents to the transformer's
    location.

    The skip weights enter linearly, so the dominant linear dependence on the
    parents is well conditioned from the start. The MLP may see other inputs
    than the parents (`SparseTriangularMap`'s parent features); the skip is
    always linear in the parents themselves.
    """

    mlp: eqx.nn.MLP
    skip: Array
    location_index: int = eqx.field(static=True)

    def __call__(self, x: Array, inputs: Array | None = None) -> Array:
        inputs = x if inputs is None else inputs
        return self.mlp(inputs).at[self.location_index].add(self.skip @ x)


def _location_index(transformer, constructor, num_params):
    """Index of the conditioner output that drives the location, or `None`.
    Found by probing `constructor`."""
    from nutpie.normalizing_flow import ElementwiseTransformer

    if not isinstance(transformer, ElementwiseTransformer):
        return None
    zeros = jnp.zeros((num_params,))
    base = constructor(zeros).location()
    if base is None:
        return None
    hits = [
        k
        for k in range(num_params)
        if constructor(zeros.at[k].set(1.0)).location() != base
    ]
    if len(hits) != 1:
        raise ValueError(
            f"Expected one conditioner output to drive the location, found {hits}."
        )
    return hits[0]


class SparseTriangularMap(bijections.AbstractBijection):
    """Triangular map whose sparsity follows a given Markov blanket.

    Variable ``i`` is conditioned on its parents ``j < i`` with
    ``blanket[i, j]``, through its own small MLP. Variables are taken in index
    order; reorder with ``Sandwich(..., Permute(order))`` (see
    `make_sparse_triangular_map`).

    `inverse_and_log_det` (model space -> whitened) reads the actual parent
    values, so it is one parallel pass. `transform_and_log_det` is ancestral
    sampling: one scan over elimination levels, where ``level(i)`` is the
    longest parent chain ending at ``i``.

    The map is lower triangular, so a Gaussian precision factors as
    ``Lambda = C^T C``. A `blanket` from a symbolic Cholesky needs the reverse
    of that elimination order.

    Conditioners are bucketed by parent count (`_min_waste_buckets`) and levels
    are split into segments (`_min_waste_segments`) to limit padding.

    Args:
        key: Jax key.
        blanket: ``(dim, dim)`` boolean adjacency; symmetrized internally.
        transformer: Unconditional scalar bijection. Defaults to
            ``make_transformer``'s.
        n_buckets: Number of conditioner-width buckets, capped at the number
            of distinct parent counts.
        n_level_segments: Number of level segments, i.e. scans per sweep,
            capped at the number of levels.
        nn_width: Conditioner hidden layer width.
        nn_depth: Conditioner hidden layer depth.
        nn_activation: Conditioner activation function.
        location_skip: Wrap conditioners in `LocationSkipMlp`. Ignored if the
            transformer has no location.
        feature_degree: If given, the conditioner MLPs see features of the
            parents instead of their raw values: each coordinate through its
            fixed marginal map to a standard normal (`marginal_to_normal`),
            then the orthonormal Hermite polynomials of degree
            ``1..feature_degree`` (`hermite_features`). The location skip
            stays linear in the raw parents, so this needs `location_skip`.
            The marginal maps start at the identity; set them with
            `with_marginal_maps`.
    """

    shape: tuple[int, ...]
    n_levels: int
    conditioners: tuple[eqx.nn.MLP | LocationSkipMlp, ...]
    bucket_members: tuple[Array, ...]
    bucket_parent_indices: tuple[Array, ...]
    # Layout of `transform_and_log_det`'s scan, one entry per level segment,
    # listing only the buckets with members there.
    #
    # level_segment_buckets[s]:            tuple of bucket indices active in s
    # level_segment_members[s][i]:         (levels_in_segment, width) global
    #                                      variable indices, padded with `dim`
    # level_segment_local_members[s][i]:   (levels_in_segment, width) positions
    #                                      within that bucket's ensemble
    level_segment_buckets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    level_segment_members: tuple[tuple[Array, ...], ...]
    level_segment_local_members: tuple[tuple[Array, ...], ...]
    transformer_constructor: Callable
    # Integer arrays, so not static (metadata must be hashable) and never
    # picked up as parameters by `eqx.is_inexact_array`.
    jacobian_layout: _SparseTriangularLayout
    # See `gauss_newton_factors`.
    selected_inverse: _SelectedInverseLayout
    bucket_sigma_index: tuple[Array, ...]
    # `(dim, 5)` `marginal_to_normal` parameters, `NonTrainable`, or None
    # when the conditioners see the raw parents.
    feature_params: Array | None
    feature_degree: int | None = eqx.field(static=True)
    cond_shape = None

    def __init__(
        self,
        key,
        *,
        blanket: ArrayLike,
        transformer: bijections.AbstractBijection | None = None,
        n_buckets: int = 8,
        n_level_segments: int = 8,
        nn_width: int = 16,
        nn_depth: int = 1,
        nn_activation: Callable = jax.nn.gelu,
        location_skip: bool = True,
        feature_degree: int | None = None,
    ):
        blanket = np.asarray(blanket, dtype=bool)
        if blanket.ndim != 2 or blanket.shape[0] != blanket.shape[1]:
            raise ValueError(
                f"blanket must be a square matrix, got shape {blanket.shape}."
            )
        dim = blanket.shape[0]

        if transformer is None:
            from nutpie.normalizing_flow import make_transformer

            transformer = make_transformer(
                affine_transformer=False,
                asymmetric_transformer=False,
                contract_transformer=2,
                # log_gamma_bounds=(-1, 1),
            )
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported."
            )

        blanket = blanket | blanket.T

        # Keep only edges from earlier to later indices.
        strictly_lower = np.tril(np.ones((dim, dim), dtype=bool), k=-1)
        parent_mask = blanket & strictly_lower

        n_parents = parent_mask.sum(axis=1)

        # Padded slots hold `dim`, which reads a zero appended to the input.
        # Sliced to each bucket's own width below.
        max_parents = int(n_parents.max(initial=0))
        parent_indices = np.full((dim, max_parents), dim, dtype=np.int32)
        for k in range(dim):
            idx = np.flatnonzero(parent_mask[k])
            parent_indices[k, : len(idx)] = idx

        # level(i): longest parent chain ending at i.
        level = np.zeros(dim, dtype=np.int64)
        for k in range(dim):
            parents_k = np.flatnonzero(parent_mask[k])
            level[k] = 0 if parents_k.size == 0 else int(level[parents_k].max()) + 1
        n_levels = int(level.max()) + 1

        n_distinct = len(np.unique(n_parents))
        n_buckets_eff = min(n_buckets, dim, n_distinct)
        bucket_of = _min_waste_buckets(n_parents, n_buckets_eff)

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        location_index = (
            _location_index(transformer, constructor, num_params)
            if location_skip
            else None
        )
        if feature_degree is not None and location_index is None:
            raise ValueError(
                "feature_degree needs the location skip (location_skip=True and "
                "a transformer with a location)."
            )
        features_per_parent = 1 if feature_degree is None else feature_degree

        def make_net(key, in_size):
            mlp = eqx.nn.MLP(
                in_size=in_size * features_per_parent,
                out_size=num_params,
                width_size=nn_width,
                depth=nn_depth,
                activation=nn_activation,
                key=key,
            )
            if location_index is None:
                return mlp
            # Zero init: the skip is linear, so a random start gains nothing.
            return LocationSkipMlp(mlp, jnp.zeros((in_size,)), location_index)

        def net_cost(in_size):
            """Rough cost of one conditioner evaluation, to weight padding."""
            in_size = in_size * features_per_parent
            if nn_depth == 0:
                return max(in_size, 1) * num_params
            return (
                max(in_size, 1) * nn_width
                + max(nn_depth - 1, 0) * nn_width**2
                + nn_width * num_params
            )

        keys = jax.random.split(key, max(n_buckets_eff, 1))

        conditioners = []
        bucket_members = []
        bucket_parent_indices = []
        bucket_net_costs = []
        # Per bucket and level: members, as global and as ensemble indices.
        bucket_members_by_level = []
        bucket_local_members_by_level = []

        for b in range(n_buckets_eff):
            members_b = np.flatnonzero(bucket_of == b)
            bucket_size_b = len(members_b)
            max_parents_b = int(n_parents[members_b].max(initial=0))

            net_keys = jax.random.split(keys[b], bucket_size_b)
            conditioners.append(
                eqx.filter_vmap(
                    lambda k, mp=max_parents_b: make_net(k, mp),
                    axis_size=bucket_size_b,
                )(net_keys)
            )
            bucket_members.append(members_b.astype(np.int32))
            bucket_parent_indices.append(
                parent_indices[members_b][:, :max_parents_b].astype(np.int32)
            )
            bucket_net_costs.append(net_cost(max_parents_b))

            local_of_global = np.zeros(dim, dtype=np.int32)
            local_of_global[members_b] = np.arange(bucket_size_b, dtype=np.int32)

            levels_b = level[members_b]
            members_at_level = [members_b[levels_b == lvl] for lvl in range(n_levels)]
            bucket_members_by_level.append(members_at_level)
            bucket_local_members_by_level.append(
                [local_of_global[idx] for idx in members_at_level]
            )

        # Size each bucket per level segment, and skip it where it is empty.
        level_bucket_counts = np.array(
            [
                [len(bucket_members_by_level[b][lvl]) for b in range(n_buckets_eff)]
                for lvl in range(n_levels)
            ],
            dtype=np.int64,
        ).reshape(n_levels, n_buckets_eff)
        segments = _min_waste_segments(
            level_bucket_counts,
            n_level_segments,
            weights=np.asarray(bucket_net_costs, dtype=np.float64),
        )

        level_segment_buckets = []
        level_segment_members = []
        level_segment_local_members = []
        for start, stop in segments:
            widths = level_bucket_counts[start:stop].max(axis=0)
            active_buckets = np.flatnonzero(widths > 0)
            members_of_segment = []
            local_members_of_segment = []
            for b in active_buckets:
                seg_members = np.full((stop - start, widths[b]), dim, dtype=np.int32)
                seg_local = np.zeros((stop - start, widths[b]), dtype=np.int32)
                for row, lvl in enumerate(range(start, stop)):
                    idx = bucket_members_by_level[b][lvl]
                    seg_members[row, : len(idx)] = idx
                    seg_local[row, : len(idx)] = bucket_local_members_by_level[b][lvl]
                members_of_segment.append(jnp.asarray(seg_members))
                local_members_of_segment.append(jnp.asarray(seg_local))
            level_segment_buckets.append(tuple(int(b) for b in active_buckets))
            level_segment_members.append(tuple(members_of_segment))
            level_segment_local_members.append(tuple(local_members_of_segment))

        self.level_segment_buckets = tuple(level_segment_buckets)
        self.level_segment_members = tuple(level_segment_members)
        self.level_segment_local_members = tuple(level_segment_local_members)

        self.conditioners = tuple(conditioners)
        self.transformer_constructor = constructor
        self.bucket_members = tuple(jnp.asarray(m) for m in bucket_members)
        self.bucket_parent_indices = tuple(
            jnp.asarray(m) for m in bucket_parent_indices
        )
        self.n_levels = n_levels
        self.shape = (dim,)
        self.feature_degree = feature_degree
        self.feature_params = (
            None
            if feature_degree is None
            else NonTrainable(jnp.zeros((dim, 5)))
        )

        self.jacobian_layout = _build_layout(
            self.bucket_members,
            self.bucket_parent_indices,
            level,
            dim,
            n_level_segments,
        )

        # For `gauss_newton_factors`: per bucket, the store index of each
        # member's `Sigma` block over `[i, parents...]`; padding -> sentinel.
        self.selected_inverse, store_index, sentinel = _build_selected_inverse(
            self.jacobian_layout.edge_child_index,
            self.jacobian_layout.edge_parent_index,
            dim,
        )
        bucket_sigma_index = []
        for members, parent_indices in zip(bucket_members, bucket_parent_indices):
            k = parent_indices.shape[1]
            index = np.full((len(members), k + 1, k + 1), sentinel, np.int32)
            for local, variable in enumerate(members):
                block = [int(variable)] + [int(p) for p in parent_indices[local]]
                for a, u in enumerate(block):
                    for b, v in enumerate(block):
                        if u < dim and v < dim:
                            index[local, a, b] = store_index(u, v)
            bucket_sigma_index.append(jnp.asarray(index))
        self.bucket_sigma_index = tuple(bucket_sigma_index)

    def with_marginal_maps(self, params) -> "SparseTriangularMap":
        """Set the marginal map of every coordinate, ``(dim, 5)`` parameters
        of `marginal_to_normal` (e.g. from `fit_marginal_maps`)."""
        if self.feature_degree is None:
            raise ValueError("The map has no parent features (feature_degree).")
        dim = self.shape[0]
        params = jnp.asarray(params)
        if params.shape != (dim, 5):
            raise ValueError(
                f"Expected params of shape ({dim}, 5), got {params.shape}."
            )
        return eqx.tree_at(lambda m: m.feature_params, self, NonTrainable(params))

    def mlp_inputs(self, parents, parent_indices):
        """What a conditioner's MLP sees: the parent features, or None if it
        sees the raw `parents`. `parent_indices` are the parents' variables
        (``dim`` for padded slots, which read zero)."""
        if self.feature_degree is None:
            return None
        dim = self.shape[0]
        # Padded slots use the identity map, and their features are zeroed.
        params = jnp.concatenate(
            [self.feature_params, jnp.zeros((1, 5), self.feature_params.dtype)]
        )[parent_indices]
        u, _ = marginal_to_normal(params, parents)
        real = (parent_indices < dim)[:, None]
        features = jnp.where(real, hermite_features(u, self.feature_degree), 0.0)
        return features.reshape(-1)

    def _condition(self, net, parents, parent_indices):
        """Transformer parameters from one conditioner at its parents."""
        inputs = self.mlp_inputs(parents, parent_indices)
        return net(parents) if inputs is None else net(parents, inputs)

    def _flat_params_to_transformer(self, params: Array):
        """``(n, num_params)`` params -> vmapped transformer."""
        transformer = eqx.filter_vmap(self.transformer_constructor)(params)
        return bijections.Vmap(transformer, in_axes=eqx.if_array(0))

    def inverse_and_log_det(self, y, condition=None):
        dim = self.shape[0]
        y_padded = jnp.concatenate([y, jnp.zeros((1,), dtype=y.dtype)])
        x = jnp.zeros((dim,), dtype=y.dtype)
        log_det = jnp.zeros(())
        for bucket in range(len(self.conditioners)):
            members = self.bucket_members[bucket]
            parent_indices = self.bucket_parent_indices[bucket]
            params = eqx.filter_vmap(self._condition)(
                self.conditioners[bucket], y_padded[parent_indices], parent_indices
            )
            transformer = self._flat_params_to_transformer(params)
            x_bucket, logdet_bucket = transformer.inverse_and_log_det(y[members])
            x = x.at[members].set(x_bucket)
            log_det = log_det + logdet_bucket
        return x, log_det

    def inverse_gradient_and_val(self, draw, grad, logp, *, parent_scores=False):
        """``(x, grad_x, logp - log_det)``, plus the per-bucket parent scores
        (see `parent_scores`) with `parent_scores`, at little extra cost."""

        def inverse_wrapper(y):
            x, log_det, *aux = self.inverse_and_log_det_and_jacobian(
                y, parent_scores=parent_scores
            )
            return log_det, (x, *aux)

        (log_det, aux), log_det_grad = jax.value_and_grad(
            inverse_wrapper, has_aux=True
        )(draw)
        x, bucket_jacobian_rows, jacobian_diagonal, *scores = aux

        edge_values = _flatten_edge_values(bucket_jacobian_rows, self.jacobian_layout)

        grad_x = _solve_triangular_sparse(
            edge_values, jacobian_diagonal, self.jacobian_layout, grad - log_det_grad
        )
        return x, grad_x, logp - log_det, *scores

    def parent_scores(self, y):
        """Per bucket ``(bucket_size, max_parents)``: the parent scores
        ``d/dy_j log q(y_i | y_pa)`` at fixed ``y_i``, zero on padded slots.

        Their mean square over draws from ``q`` is the squared Fisher speed of
        the conditional ``q(. | y_pa)`` along parent ``j`` (see
        `notes/flow_fisher_regularizer.md`, option B).
        """
        return self.inverse_and_log_det_and_jacobian(y, parent_scores=True)[-1]

    def gauss_newton_factors(
        self, y, grad, cholesky_jitter=None, fisher_regularization=None
    ):
        """Per-draw factors of the exact Gauss-Newton blocks of the Fisher
        residual ``r = x + w``, with ``J^T w = grad - grad_y log_det``.

        Returns per bucket ``V`` of shape ``(bucket_size, k + 2, n_params)``,
        ``n_params`` in `ravel_pytree` order, such that variable ``i``'s block
        is ``sum_draws V_i^T V_i``. With ``a = dx_i/dtheta_i``, ``B`` the
        Jacobian of ``q = -(d log_det_i/dy) - (dx_i/dy) w_i`` over
        ``{i} + P(i)``, ``K = (J^T J)^{-1} = L L^T`` on that set and
        ``c = e_0 / delta_i`` (see `notes/lm_derivatives.md`):

            V = [ sqrt(1 - |L^{-1} c|^2) a ;  L^T B + (L^{-1} c) a ].

        ``K`` is read off the selected inverse of ``J^T J``, and for badly
        conditioned ``J`` rounding can make it numerically indefinite. With
        a `cholesky_jitter`, a failed Cholesky is retried on
        ``K + cholesky_jitter * max(diag K) I``.

        With a `fisher_regularization` ``lambda``, ``V`` gets ``k`` more rows,
        ``sqrt(lambda)`` times the Jacobian of `parent_scores`, for the
        regularizer residuals. Each only depends on its own conditioner.
        """
        dim = self.shape[0]

        def log_det_with_aux(y):
            _, log_det, rows, diagonal = self.inverse_and_log_det_and_jacobian(y)
            return log_det, (rows, diagonal)

        (_, (rows, diagonal)), log_det_grad = jax.value_and_grad(
            log_det_with_aux, has_aux=True
        )(y)
        edge_values = _flatten_edge_values(rows, self.jacobian_layout)
        w = _solve_triangular_sparse(
            edge_values, diagonal, self.jacobian_layout, grad - log_det_grad
        )
        store = _selected_inverse(edge_values, diagonal, self.selected_inverse)

        y_padded = jnp.concatenate([y, jnp.zeros((1,), y.dtype)])

        def factor(net, parents, parent_indices, real, value, w_i, delta_i, sigma):
            arrays, static = eqx.partition(net, eqx.is_inexact_array)
            flat, unravel = jax.flatten_util.ravel_pytree(arrays)

            def local(flat):
                net = eqx.combine(unravel(flat), static)

                def element(parents, value):
                    transformer = self.transformer_constructor(
                        self._condition(net, parents, parent_indices)
                    )
                    return transformer.inverse_and_log_det(value)

                x_i, _ = element(parents, value)
                (dx_dp, dx_dy), (dl_dp, dl_dy) = jax.jacrev(element, argnums=(0, 1))(
                    parents, value
                )
                q_own = -dl_dy - dx_dy * w_i
                q_parents = -dl_dp - dx_dp * w_i
                outputs = [x_i[None], q_own[None], q_parents]
                if fisher_regularization is not None:
                    # `parent_scores`, from the same derivatives
                    outputs.append(dl_dp - x_i * dx_dp)
                return jnp.concatenate(outputs)

            # (k + 2, n_params), plus k score rows if regularized
            jac = jax.jacrev(local)(flat)
            n_block = real.shape[0]
            a = jac[0]
            # Padded parent slots read a constant, not a variable: no row.
            B = jac[1 : n_block + 1] * real[:, None]
            # Identity on padded slots keeps `K` factorable.
            pad = ~real
            K = jnp.where(pad[:, None] | pad[None, :], jnp.eye(real.shape[0]), sigma)
            L = jnp.linalg.cholesky(K)
            if cholesky_jitter is not None:
                jitter = cholesky_jitter * jnp.max(jnp.diagonal(K))
                L_jittered = jnp.linalg.cholesky(K + jitter * jnp.eye(real.shape[0]))
                L = jnp.where(jnp.all(jnp.isfinite(L)), L, L_jittered)
            c = jnp.zeros(real.shape[0], a.dtype).at[0].set(1.0 / delta_i)
            Linv_c = jax.scipy.linalg.solve_triangular(L, c, lower=True)
            s = jnp.sqrt(jnp.maximum(1.0 - Linv_c @ Linv_c, 0.0))
            rows = [(s * a)[None], L.T @ B + jnp.outer(Linv_c, a)]
            if fisher_regularization is not None:
                scores = jac[n_block + 1 :] * real[1:, None]
                rows.append(jnp.sqrt(fisher_regularization) * scores)
            return jnp.concatenate(rows)

        factors = []
        for bucket, conditioner in enumerate(self.conditioners):
            members = self.bucket_members[bucket]
            parent_indices = self.bucket_parent_indices[bucket]
            real = jnp.concatenate(
                [
                    jnp.ones((members.shape[0], 1), bool),
                    parent_indices < dim,
                ],
                axis=1,
            )
            factors.append(
                eqx.filter_vmap(factor)(
                    conditioner,
                    y_padded[parent_indices],
                    parent_indices,
                    real,
                    y[members],
                    w[members],
                    diagonal[members],
                    store[self.bucket_sigma_index[bucket]],
                )
            )
        return factors

    def inverse_and_log_det_and_jacobian(
        self, y, condition=None, *, parent_scores=False
    ):
        """Parallel y -> x pass returning ``x``, ``log|det J|``, and ``J`` as
        per-bucket rows ``(bucket_size, max_parents)`` plus its diagonal.

        With `parent_scores`, also returns the per-bucket parent scores (see
        `parent_scores`), from a second pullback of the same pass."""
        (dim,) = self.shape
        y_padded = jnp.concatenate([y, jnp.zeros((1,), y.dtype)])

        def differentiate_one_variable(
            conditioner, parent_values, parent_indices, own_value
        ):
            @jax.profiler.annotate_function
            def transform_element(parents, value):
                params = self._condition(conditioner, parents, parent_indices)
                transformer = self.transformer_constructor(params)
                return transformer.inverse_and_log_det(value)

            (x_i, log_det_i), pull = jax.vjp(
                transform_element, parent_values, own_value
            )
            one, zero = jnp.ones_like(x_i), jnp.zeros_like(x_i)
            parent_derivatives, own_derivative = pull((one, zero))
            if not parent_scores:
                return x_i, log_det_i, parent_derivatives, own_derivative
            # d/dy_pa (log_det_i - x_i**2 / 2) at fixed y_i
            log_det_parent_derivatives, _ = pull((zero, one))
            scores = jnp.where(
                parent_indices < dim,
                log_det_parent_derivatives - x_i * parent_derivatives,
                0.0,
            )
            return x_i, log_det_i, parent_derivatives, own_derivative, scores

        x = jnp.zeros((dim,), y.dtype)
        jacobian_diagonal = jnp.zeros((dim,), y.dtype)
        log_det = jnp.zeros((), y.dtype)
        bucket_jacobian_rows = []
        bucket_scores = []

        for bucket in range(len(self.conditioners)):
            members = self.bucket_members[bucket]
            parent_indices = self.bucket_parent_indices[bucket]

            bucket_x, bucket_log_det, bucket_rows, bucket_diagonal, *scores = (
                eqx.filter_vmap(differentiate_one_variable)(
                    self.conditioners[bucket],
                    y_padded[parent_indices],
                    parent_indices,
                    y[members],
                )
            )

            x = x.at[members].set(bucket_x)
            jacobian_diagonal = jacobian_diagonal.at[members].set(bucket_diagonal)
            log_det = log_det + jnp.sum(bucket_log_det)
            bucket_jacobian_rows.append(bucket_rows)
            bucket_scores.extend(scores)

        out = (x, log_det, tuple(bucket_jacobian_rows), jacobian_diagonal)
        return (*out, tuple(bucket_scores)) if parent_scores else out

    def transform_and_log_det(self, x, condition=None):
        """Ancestral x -> y pass: one scan per level segment."""
        dim = self.shape[0]

        def make_step(buckets):
            def step(carry, level_data):
                y, log_det = carry
                members_of_level, local_members_of_level = level_data
                y_next = y
                for bucket, members, local_members in zip(
                    buckets, members_of_level, local_members_of_level
                ):
                    # Parents come from `y` at the start of the level. Padding
                    # (`members == dim`) is dropped by the scatter and `where`.
                    parent_idx = self.bucket_parent_indices[bucket][local_members]
                    parents = y.at[parent_idx].get(mode="fill", fill_value=0.0)
                    conditioner_group = jax.tree.map(
                        lambda leaf: (
                            leaf[local_members] if eqx.is_array(leaf) else leaf
                        ),
                        self.conditioners[bucket],
                    )

                    def transform_element(net, parent_values, indices, value):
                        params = self._condition(net, parent_values, indices)
                        transformer = self.transformer_constructor(params)
                        return transformer.transform_and_log_det(value)

                    y_group, log_det_group = eqx.filter_vmap(transform_element)(
                        conditioner_group,
                        parents,
                        parent_idx,
                        x.at[members].get(mode="fill", fill_value=0.0),
                    )
                    y_next = y_next.at[members].set(
                        y_group, mode="drop", unique_indices=True
                    )
                    log_det = log_det + jnp.sum(
                        jnp.where(members != dim, log_det_group, 0.0)
                    )
                return (y_next, log_det), None

            return step

        y = x
        log_det = jnp.zeros((), x.dtype)
        for buckets, members, local_members in zip(
            self.level_segment_buckets,
            self.level_segment_members,
            self.level_segment_local_members,
        ):
            (y, log_det), _ = jax.lax.scan(
                make_step(buckets), (y, log_det), (members, local_members)
            )
        return y, log_det
