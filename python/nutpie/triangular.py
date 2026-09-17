from paramax import NonTrainable
from flowjax.utils import get_ravelled_pytree_constructor
import jax
from jax.typing import ArrayLike
from typing import Callable, ClassVar
import numpy as np
import jax.numpy as jnp
from jax import Array
from flowjax import bijections
import equinox as eqx



_N_LEVEL_SEGMENTS = 8


def _min_waste_segments(
    counts: np.ndarray, n_segments: int, weights: np.ndarray | None = None
) -> list:
    """Split `counts` into at most `n_segments` *contiguous* runs, minimizing
    ``sum(run_max * len(run))`` -- the number of padded slots when each run is
    stored as one rectangular array of its own width.

    This is `_min_waste_buckets`' objective (the two differ by ``sum(counts)``,
    a constant) under one extra constraint: the runs must be contiguous in the
    *given* order, not in sorted order. Elimination levels cannot be reordered,
    since back substitution needs level ``l + 1`` resolved before level ``l``,
    so the sortedness that lets `_min_waste_buckets` read a group's maximum off
    its last element is unavailable and the maximum is carried explicitly.

    Splitting matters whenever the per-level edge counts are skewed, which a
    single variable that parents many others is enough to cause: one level then
    sizes the array for all of them.

    `counts` may also be ``(n, n_groups)``, for the case where each level is
    stored as one rectangular array *per group* -- one per conditioner bucket
    in `SparseTriangularMap.transform_and_log_det`. Each group is then padded
    to its own per-run maximum and the run's cost is the weighted sum over
    groups, ``len(run) * sum_g weights[g] * run_max[g]``. `weights` is the
    relative cost of one padded slot in each group (a bucket whose conditioner
    reads 200 parents wastes far more per slot than one that reads 1), and
    defaults to uniform.

    Returns:
        List of ``(start, stop)`` index pairs covering ``range(len(counts))``.
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
    """Partition `counts` into at most `n_buckets` groups, minimizing the
    total padding waste ``sum(group_max - value)`` that results from padding
    every value in a group up to that group's own maximum.

    This is exactly the cost `SparseTriangularMap` cares about when sizing
    conditioner-network buckets by parent count: it's the number of wasted
    (zero-padded) conditioner input columns, summed over all variables. The
    optimal groups are always contiguous ranges of the *sorted* values
    (grouping a value with smaller ones it isn't padded down to never
    helps), so this is a small, exact dynamic program -- no need for an
    approximate heuristic or an external clustering library, and no need to
    reach for the general (and here unnecessary) machinery of optimal
    1D-clustering algorithms: with `n` items and `n_buckets` groups it's
    O(n^2 * n_buckets), which is negligible at the sizes this is used for
    (this runs once, at construction time).

    Returns:
        `(len(counts),)` int array giving each item's bucket index (0 is
        the bucket containing the smallest values), in the same order as
        `counts`.
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
    """Static sparsity structure of the Jacobian. Built once, outside jit.

    `inverse_gradient_and_val` only ever needs ``J^T w = rhs``, resolved by
    back-substitution: levels last-to-first, each variable summing
    contributions from its already-resolved children. That is what the
    ``level_*`` arrays below are grouped for -- by the level of each edge's
    *parent*.

    Differentiating that solve needs the other direction too: the cotangent of
    ``A^-1 rhs`` involves ``A^-T``, i.e. forward substitution on ``J`` itself,
    where a variable sums over its *parents* and levels run first-to-last. The
    ``child_level_*`` arrays are the same edges regrouped by the level of each
    edge's child, which is what that traversal gathers on. Both groupings are
    pure index arrays built once, outside jit.
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
    # The level arrays are split into contiguous segments of levels, each
    # stored at its own width (see `_min_waste_segments`). A single array over
    # all levels would be padded to the widest level's edge count, and one
    # variable that parents many others makes that the whole dimension.
    #
    # Segments are in level order, so a sweep runs one scan per segment,
    # visiting them last-to-first for back substitution and first-to-last for
    # its transpose. Each entry below is a tuple with one array per segment.
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
    n_level_segments=_N_LEVEL_SEGMENTS,
):
    """Derive the static edge layout from the model's bucket and level data.

    Edges are grouped by the level of their *parent*, not their child,
    because the solve this feeds (`_solve_triangular_sparse`) is the
    transpose of a forward triangular solve: it resolves each variable from
    the contributions of its children, which requires visiting levels
    last-to-first (see that function's docstring).
    """
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
        """Per-segment (edge index, target slot) arrays at each segment's own
        width, together with the edges' owning level."""
        by_level = [
            np.flatnonzero(level_of_edge == level) for level in range(n_levels)
        ]
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

    # Segment the level range so that one wide level does not size the arrays
    # for all of them; both groupings get their own segmentation, since their
    # per-level edge counts are unrelated.
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
    """Ragged per-bucket derivative rows -> flat (n_edges + 1,) edge values.

    The loop runs once per bucket (a handful of iterations) and is pure
    gather; it sits outside the level scan.
    """
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
    """One triangular solve as a scan over elimination levels.

    Both directions are the same sweep with the edges grouped differently, so
    they share this body. `edge_endpoint_index` says which endpoint of each
    edge supplies the already-solved value (the child for back substitution,
    the parent for forward substitution), and the level arrays say which
    edges land on which level and in whose slot.

    The level arrays arrive as one array per contiguous segment of levels, each
    at its own width, so this runs one scan per segment rather than one over
    all levels at the widest level's width. Segments are visited in the sweep's
    own direction, and `reverse` applies within each.
    """
    max_level_size = layout.max_level_size

    # Sentinel row: rhs 0 and diagonal 1, so padded lanes stay finite. Their
    # values are discarded, but a NaN would still poison a cotangent.
    rhs_padded = jnp.concatenate([rhs, jnp.zeros((1,), rhs.dtype)])
    diagonal_padded = jnp.concatenate([jacobian_diagonal, jnp.ones((1,), rhs.dtype)])

    @jax.profiler.annotate_function
    def eliminate_level(solution, level_data):
        members, edge_indices, target_slots = level_data
        # The endpoint each edge reads from always sits at an already-visited
        # level, by construction of the level numbering.
        solution_padded = jnp.concatenate([solution, jnp.zeros((1,), rhs.dtype)])
        solved_values = solution_padded[edge_endpoint_index[edge_indices]]
        edge_contributions = edge_values[edge_indices] * solved_values

        # Sum contributions into the slot of the variable they belong to.
        # Padding edges target slot `max_level_size`, which is out of bounds
        # and dropped.
        neighbour_sum = (
            jnp.zeros((max_level_size,), rhs.dtype)
            .at[target_slots]
            .add(edge_contributions, mode="drop")
        )

        updated = (rhs_padded[members] - neighbour_sum) / diagonal_padded[members]
        # Each variable belongs to exactly one level, so the in-bounds
        # indices here are unique.
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
    """Solve ``J^T w = rhs`` for the sparse triangular Jacobian ``J``.

    `inverse_gradient_and_val` only ever needs this transposed direction
    (pulling a model-space cotangent back through the parallel m -> w map, see
    that method's docstring), which is back substitution: levels last-to-first,
    each variable summing over its *children*, all of which sit at strictly
    later levels and so are already solved.

    Wrapped in `jax.lax.custom_linear_solve` rather than left to autodiff.
    Differentiating the scan directly stacks one carry -- a full ``(dim,)``
    solution -- per elimination level, so the tape is ``n_levels * dim`` per
    draw. That is not a corner case: `SparseTriangularMap`'s level grouping
    degenerates to one level per variable for a *dense* blanket and equally for
    a banded one, so ``n_levels == dim`` is the common case and the tape is
    quadratic in the dimension. Measured on a 1600-dim banded model it was a
    156 MB ``f64[n_levels, batch, dim]`` buffer dominating the whole step.
    """

    def matvec(w):
        """``J^T w``, from the same edge list -- one scatter, no sweep."""
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


class SumLinearAndMlp(eqx.Module):
    linear: eqx.nn.Linear
    mlp: eqx.nn.MLP

    def __init__(
        self,
        linear: eqx.nn.Linear,
        mlp: eqx.nn.MLP,
    ):
        super().__init__()
        self.linear = linear
        self.mlp = mlp

    def __call__(self, x: Array) -> Array:
        linear_out = self.linear(x)
        mlp_out = self.mlp(x)
        return linear_out + mlp_out


class SparseTriangularMap(bijections.AbstractBijection):
    """Triangular map with a caller-specified sparsity pattern.

    A standard masked autoregressive flow (see e.g.
    ``flowjax.bijections.MaskedAutoregressive``) lets every transformed
    variable depend on *all* variables preceding it. If the factorization of
    the target distribution is (approximately) known -- for instance because
    the Markov blanket of each variable has already been identified -- most
    of those dependencies are unnecessary. This bijection instead gives
    every variable its own small conditioner network that only ever sees the
    variables in its Markov blanket that precede it. Because non-parent
    variables never reach a variable's conditioner, the resulting Jacobian is
    exactly triangular with the specified sparsity pattern (rather than
    merely triangular, as for a dense MADE-style flow), and the conditioner
    networks can be made much smaller than a dense autoregressive
    conditioner.

    This bijection treats variable ``i`` as preceding variable ``j`` whenever
    ``i < j``, i.e. it assumes the variables are already given in the
    desired order. To use a different variable ordering, wrap it as
    ``bijections.Sandwich(SparseTriangularMap(...), bijections.Permute(order))``
    (see `make_sparse_triangular_map`).

    Which direction is `inverse_and_log_det` and which is
    `transform_and_log_det` is not an arbitrary choice, and it is not merely
    a performance question. `blanket` is a statement about how the density
    of the *model-space* variable factorizes, ``p(m) = prod_i p(m_i |
    m_parents(i))`` -- the same role a sparse precision matrix plays for a
    Gaussian: sparse ``Lambda`` gives a sparse, direct whitening map ``w = C
    m`` (a plain matrix-vector product using the true blanket entries of the
    actual data ``m``), whereas the reverse map ``m = C^{-1} w`` solves a
    triangular system and is generally dense/sequential, because ``w`` is
    noise and the blanket was never a statement about how noise combines. A
    conditioner only "uses the Markov blanket of ``m``" if it is literally a
    function of ``m``'s actual parent values; conditioning on the
    corresponding entries of ``w`` instead would still be invertible, but
    would no longer correspond to anything about the density we were told to
    respect.

    Note the triangle convention that ``C`` implies, since it is easy to get
    backwards. Here ``C`` is *lower* triangular (variable ``i`` sees only
    ``j < i``), so whitening a Gaussian means factorizing its precision as
    ``Lambda = C^T C`` -- a reverse (UL) Cholesky, not the usual ``Lambda =
    L L^T``. The two have different fill patterns: the fill of ``C`` for a
    given elimination order equals the fill of ``L`` for the *reversed*
    order. So a `blanket` obtained by symbolic factorization (CHOLMOD/AMD or
    similar) must be paired with the reverse of the elimination order it was
    computed for -- see `make_sparse_triangular_map`'s ``order`` argument.
    Getting this wrong yields a pattern that silently cannot represent the
    target at all, rather than one that merely fits it badly (though it is
    invisible for patterns that are fill-free in both directions, such as a
    banded/tridiagonal one).

    Concretely: `inverse_and_log_det` takes the model-space point ``m`` (or,
    when sandwiched with a `bijections.Permute`, a reindexing of it) and
    computes every conditioner directly from ``m`` in one parallel pass --
    this is the "evaluate the density" direction, and it is also what
    nutpie's transform-adapted NUTS sampler calls at every leapfrog step (see
    ``nuts-rs``'s ``Transformation::inv_transform_normalize`` and
    ``transform_adapter.inverse_gradient_and_val``, both of which pass in the
    untransformed/model-space position). `transform_and_log_det` is
    ancestral sampling from the whitened point back to ``m``: it must
    resolve ``m`` sequentially (via `jax.lax.scan`), since each conditioner
    needs the already-resolved *model-space* parents, not the noise.

    That sequential resolution doesn't have to go variable by variable,
    though: variables whose parents are all already resolved are mutually
    independent and can be resolved together. `transform_and_log_det`
    exploits this by grouping variables into "elimination levels" --
    ``level(i)`` is the length of the longest parent-chain ending at ``i``,
    so level 0 is every variable with no parents, level 1 is every variable
    whose parents are all in level 0, and so on -- and scanning over levels
    (each processed as one vmapped batch) rather than over individual
    variables. The number of levels is the DAG's critical-path depth, the
    minimum number of sequential stages any schedule could achieve; for a
    fully dense `blanket` every variable ends up in its own level and this
    degenerates to the naive per-variable scan, while a shallow/tree-like
    `blanket` can cut the sequential depth from ``dim`` down to
    ``O(log dim)`` or less. The tradeoff is that levels are padded to a
    common width, so this trades sequential steps for total work and is a
    net win only when levels are reasonably balanced. The padding is kept
    to what the imbalance really costs by splitting the level range into
    contiguous segments, each bucket sized per segment and skipped over the
    segments where it has no members (see `_min_waste_segments`).

    Separately, conditioner networks are grouped into `n_buckets` buckets by
    parent count (see `_min_waste_buckets`), each with its own (smaller)
    input width, rather than every variable's conditioner paying for the
    input width the single worst-connected variable needs. This matters
    independently of the level grouping: a handful of variables with large
    parent counts is common, and without bucketing every other variable's
    conditioner -- whether processed in `inverse_and_log_det`'s single pass
    or within one level of `transform_and_log_det`'s scan -- would pay for
    that width too.

    Args:
        key: Jax key.
        blanket: A ``(dim, dim)`` array, convertible to boolean.
            ``blanket[i, j]`` being truthy means ``j`` is used to
            parameterize the transform of ``i``, provided ``j < i``. The
            matrix is symmetrized internally, so it is fine to pass e.g. an
            undirected Markov-blanket adjacency matrix.
        transformer: Unconditional bijection with shape ``()``, applied
            elementwise to each variable. Defaults to this module's
            standard elementwise transformer, see ``make_transformer``.
        n_buckets: Number of conditioner-width buckets, see above. Capped
            automatically at the number of distinct parent counts.
        nn_width: Conditioner hidden layer width.
        nn_depth: Conditioner hidden layer depth.
        nn_activation: Conditioner activation function.
    """

    shape: tuple[int, ...]
    n_levels: int
    conditioners: tuple[eqx.nn.MLP, ...]
    bucket_members: tuple[Array, ...]
    bucket_parent_indices: tuple[Array, ...]
    # Layout of `transform_and_log_det`'s level scan, one entry per contiguous
    # segment of levels (see `_min_waste_segments`). Within a segment, only the
    # buckets that actually have members there are listed, each padded to its
    # own width over that segment's levels -- rather than every bucket being
    # evaluated at every level, at its global maximum level occupancy.
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
    # A normal pytree field, not `static=True`. Its leaves are index arrays, so
    # marking them static made them pytree *metadata*, which JAX compares for
    # equality -- comparing arrays yields an array, not a bool, so anything that
    # triggers that comparison raises (and it is what the "A JAX array is being
    # set as static!" warning was about). As leaves they are integer arrays, so
    # `eqx.partition(flow, eqx.is_inexact_array)` still keeps them out of the
    # parameters and they stay compile-time constants in practice.
    jacobian_layout: _SparseTriangularLayout
    cond_shape = None

    def __init__(
        self,
        key,
        *,
        blanket: ArrayLike,
        transformer: bijections.AbstractBijection | None = None,
        n_buckets: int = 8,
        nn_width: int = 16,
        nn_depth: int = 1,
        nn_activation: Callable = jax.nn.gelu,
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
                #log_gamma_bounds=(-1, 1),
            )
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported."
            )

        blanket = blanket | blanket.T

        # Only keep edges that point from an earlier to a later index, so
        # that the resulting transform is guaranteed to be triangular.
        strictly_lower = np.tril(np.ones((dim, dim), dtype=bool), k=-1)
        parent_mask = blanket & strictly_lower

        n_parents = parent_mask.sum(axis=1)

        # Sentinel index `dim` always reads a constant zero appended to x, so
        # unused (padded) slots never leak information. `max_parents` here
        # is the *global* max, only used to build a single padded array
        # that gets sliced down per-bucket below.
        max_parents = int(n_parents.max(initial=0))
        parent_indices = np.full((dim, max_parents), dim, dtype=np.int32)
        for k in range(dim):
            idx = np.flatnonzero(parent_mask[k])
            parent_indices[k, : len(idx)] = idx

        # Elimination levels: level(i) is the length of the longest
        # parent-chain ending at i, so all variables sharing a level are
        # mutually independent given earlier levels (see the class
        # docstring). This is the minimum possible number of sequential
        # stages for any valid schedule.
        level = np.zeros(dim, dtype=np.int64)
        for k in range(dim):
            parents_k = np.flatnonzero(parent_mask[k])
            level[k] = 0 if parents_k.size == 0 else int(level[parents_k].max()) + 1
        n_levels = int(level.max()) + 1

        # Bucket variables by parent count, so that variables with few
        # parents don't pay for the conditioner width the rare
        # many-parents variable needs (see `_min_waste_buckets`).
        n_distinct = len(np.unique(n_parents))
        n_buckets_eff = min(n_buckets, dim, n_distinct)
        bucket_of = _min_waste_buckets(n_parents, n_buckets_eff)

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )

        def make_net(key, in_size):
            key, key_linear = jax.random.split(key)
            linear = eqx.nn.Linear(in_size, num_params, key=key_linear)

            linear = eqx.tree_at(lambda l: l.weight, linear, 1e-3 * linear.weight)
            linear = eqx.tree_at(lambda l: l.bias, linear, 1e-3 * linear.bias)

            mlp = eqx.nn.MLP(
                in_size=in_size,
                out_size=num_params,
                width_size=nn_width,
                depth=nn_depth,
                activation=nn_activation,
                key=key,
            )
            return mlp  # SumLinearAndMlp(linear, mlp)

        def net_cost(in_size):
            """Rough cost of one conditioner evaluation, used to weight the
            padding waste of a bucket against the other buckets'."""
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
        # Per bucket, the bucket's members at each level, as global indices and
        # as positions within the bucket's own ensemble.
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

            # local position of each global variable index within this
            # bucket's own (bucket_size_b,)-shaped ensemble/member list, so
            # that a level's subset of this bucket can be gathered from it.
            local_of_global = np.zeros(dim, dtype=np.int32)
            local_of_global[members_b] = np.arange(bucket_size_b, dtype=np.int32)

            levels_b = level[members_b]
            members_at_level = [members_b[levels_b == lvl] for lvl in range(n_levels)]
            bucket_members_by_level.append(members_at_level)
            bucket_local_members_by_level.append(
                [local_of_global[idx] for idx in members_at_level]
            )

        # The level scan is sequential in the levels but not in the buckets, so
        # a naive layout evaluates every bucket at every level, padded to that
        # bucket's widest level. Splitting the level range into contiguous
        # segments lets each bucket be sized per segment instead -- and lets a
        # bucket be skipped entirely over the levels where it has no members at
        # all, which is the common case for the few-but-wide buckets.
        level_bucket_counts = np.array(
            [
                [len(bucket_members_by_level[b][lvl]) for b in range(n_buckets_eff)]
                for lvl in range(n_levels)
            ],
            dtype=np.int64,
        ).reshape(n_levels, n_buckets_eff)
        segments = _min_waste_segments(
            level_bucket_counts,
            _N_LEVEL_SEGMENTS,
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

        self.jacobian_layout = _build_layout(
            self.bucket_members, self.bucket_parent_indices, level, dim
        )

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to n x params_per_dim, then vmap."""
        transformer = eqx.filter_vmap(self.transformer_constructor)(params)
        return bijections.Vmap(transformer, in_axes=eqx.if_array(0))

    def inverse_and_log_det(self, y, condition=None):
        dim = self.shape[0]
        y_padded = jnp.concatenate([y, jnp.zeros((1,), dtype=y.dtype)])
        x = jnp.zeros((dim,), dtype=y.dtype)
        log_det = jnp.zeros(())
        for bucket in range(len(self.conditioners)):
            members = self.bucket_members[bucket]
            parents = y_padded[self.bucket_parent_indices[bucket]]
            params = eqx.filter_vmap(lambda net, inp: net(inp))(
                self.conditioners[bucket], parents
            )
            transformer = self._flat_params_to_transformer(params)
            x_bucket, logdet_bucket = transformer.inverse_and_log_det(y[members])
            x = x.at[members].set(x_bucket)
            log_det = log_det + logdet_bucket
        return x, log_det

    def inverse_gradient_and_val(self, draw, grad, logp):
        def inverse_wrapper(y):
            x, log_det, bucket_jacobian_rows, jacobian_diagonal = (
                self.inverse_and_log_det_and_jacobian(y)
            )
            return log_det, (x, bucket_jacobian_rows, jacobian_diagonal)

        ((log_det, (x, bucket_jacobian_rows, jacobian_diagonal)), log_det_grad) = (
            jax.value_and_grad(inverse_wrapper, has_aux=True)(draw)
        )

        edge_values = _flatten_edge_values(bucket_jacobian_rows, self.jacobian_layout)

        grad_x = _solve_triangular_sparse(
            edge_values, jacobian_diagonal, self.jacobian_layout, grad - log_det_grad
        )
        return x, grad_x, logp - log_det

    def inverse_and_log_det_and_jacobian(self, y, condition=None):
        """Parallel y -> x pass returning x, log|det J|, and the sparse J.

        `apply_transformer(params, value) -> (x_i, log_det_i)` is the scalar
        transformer in the same direction the parallel map uses.

        Returns the Jacobian as (bucket_jacobian_rows, jacobian_diagonal), where
        bucket_jacobian_rows[bucket] has shape (bucket_size, max_parents_in_bucket).
        """
        (dim,) = self.shape
        y_padded = jnp.concatenate([y, jnp.zeros((1,), y.dtype)])

        def differentiate_one_variable(conditioner, parent_values, own_value):
            @jax.profiler.annotate_function
            def transform_element(parents, value):
                params = conditioner(parents)
                transformer = self.transformer_constructor(params)
                return transformer.inverse_and_log_det(value)

            (x_i, log_det_i), (parent_derivatives, own_derivative) = jax.value_and_grad(
                transform_element, argnums=(0, 1), has_aux=True
            )(parent_values, own_value)
            return x_i, log_det_i, parent_derivatives, own_derivative

        x = jnp.zeros((dim,), y.dtype)
        jacobian_diagonal = jnp.zeros((dim,), y.dtype)
        log_det = jnp.zeros((), y.dtype)
        bucket_jacobian_rows = []

        # One iteration per bucket. Ensembles have different input widths, so they
        # cannot be merged; all iterations are independent and depth stays 1.
        for bucket in range(len(self.conditioners)):
            members = self.bucket_members[bucket]
            parent_indices = self.bucket_parent_indices[bucket]

            bucket_x, bucket_log_det, bucket_rows, bucket_diagonal = eqx.filter_vmap(
                differentiate_one_variable
            )(self.conditioners[bucket], y_padded[parent_indices], y[members])

            x = x.at[members].set(bucket_x)
            jacobian_diagonal = jacobian_diagonal.at[members].set(bucket_diagonal)
            log_det = log_det + jnp.sum(bucket_log_det)
            bucket_jacobian_rows.append(bucket_rows)

        return x, log_det, tuple(bucket_jacobian_rows), jacobian_diagonal

    def transform_and_log_det(self, x, condition=None):
        """Ancestral x -> y pass: one scan per level segment, log det included.

        The log det comes out of the same conditioner evaluations that resolve
        `y`, since the scalar transformer returns it anyway; running the
        parallel y -> x pass afterwards just to obtain it would evaluate every
        conditioner a second time.
        """
        dim = self.shape[0]

        def make_step(buckets):
            def step(carry, level_data):
                y, log_det = carry
                members_of_level, local_members_of_level = level_data
                y_next = y
                for bucket, members, local_members in zip(
                    buckets, members_of_level, local_members_of_level
                ):
                    # Parents are read from `y` as of the *start* of this level
                    # (safe -- variables in the same level never depend on each
                    # other), so buckets within a level can be processed in any
                    # order. Padding slots (`members == dim`) read zeros, and
                    # their results are discarded by the `mode="drop"` scatter
                    # and the `where` on the log det.
                    parent_idx = self.bucket_parent_indices[bucket][local_members]
                    parents = y.at[parent_idx].get(mode="fill", fill_value=0.0)
                    conditioner_group = jax.tree.map(
                        lambda leaf: leaf[local_members]
                        if eqx.is_array(leaf)
                        else leaf,
                        self.conditioners[bucket],
                    )

                    def transform_element(net, parent_values, value):
                        params = net(parent_values)
                        transformer = self.transformer_constructor(params)
                        return transformer.transform_and_log_det(value)

                    y_group, log_det_group = eqx.filter_vmap(transform_element)(
                        conditioner_group,
                        parents,
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
