import jax, jax.numpy as jnp, numpy as np, equinox as eqx
from jax import random as jr
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular


# ============================================================ pytree helpers


def tdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def tnorm(a):
    return jnp.sqrt(tdot(a, a))


def codec(block):
    """block: pytree whose inexact leaves share leading axis n (one bucket).
    Returns (flatten, unflatten, Pb) with flatten: pytree -> (n, Pb)."""
    arr, static = eqx.partition(block, eqx.is_inexact_array)
    one = jax.tree.map(lambda l: l[0], arr)
    flat0, unravel_one = ravel_pytree(one)

    def flatten(t):
        a, _ = eqx.partition(t, eqx.is_inexact_array)
        return jax.vmap(lambda mem: ravel_pytree(mem)[0])(a)

    def unflatten(A):
        return eqx.combine(jax.vmap(unravel_one)(A), static)

    return flatten, unflatten, flat0.size


# ============================================================ flow structure


def _conditioners(tree):
    return tree.bijection.bijections[0].bijections[0].inner.conditioners


def _diag_affine(tree):
    return tree.bijection.bijections[1]


def blocks_of(tree):
    """Parameter blocks: one per conditioner bucket, plus the diagonal affine
    tail unless `split_frozen` has taken it out (see `fit`'s `fit_affine`)."""
    tail = _diag_affine(tree)
    return [*_conditioners(tree), *([] if tail is None else [tail])]


def rebuild(tree, blocks):
    """Inverse of blocks_of."""
    n_conditioners = len(_conditioners(tree))
    tree = eqx.tree_at(_conditioners, tree, tuple(blocks[:n_conditioners]))
    if len(blocks) > n_conditioners:
        tree = eqx.tree_at(_diag_affine, tree, blocks[n_conditioners])
    return tree


def split_frozen(params, fit_affine=False):
    """Split off the diagonal affine tail, unless `fit_affine`.

    It is (near-)redundant with each conditioner's own output shift and scale,
    so fitting it mostly adds coupling *between* blocks that the
    block-diagonal preconditioner cannot see. `make_flow` initializes it at the
    per-variable Gaussian optimum, and the transformers can absorb any later
    drift.

    Returns ``(trainable, frozen)``. When frozen, `trainable` has the tail
    replaced by ``None``, so every pytree-wide operation in `step` skips it;
    when fitted, `frozen` is ``None`` and `trainable` is `params` unchanged.
    """
    if fit_affine:
        return params, None
    return eqx.tree_at(_diag_affine, params, None), _diag_affine(params)


def merge_frozen(trainable, frozen):
    """Inverse of split_frozen."""
    if frozen is None:
        return trainable
    return eqx.tree_at(_diag_affine, trainable, frozen, is_leaf=lambda x: x is None)


# ============================================================ sub-blocking


def mlp_unit_labels(one):
    """Depth-1 equinox MLP: label each coordinate by hidden unit, so a sub-block
    is never a single layer (which conditions badly when layer scales differ).
    A `LocationSkipMlp`'s skip weights get a label of their own, after every
    unit's, so they stay together. Returns label arrays in jax.tree.leaves
    order."""
    from nutpie.triangular import LocationSkipMlp

    if isinstance(one, LocationSkipMlp):
        n_units = one.mlp.layers[0].weight.shape[0]
        return mlp_unit_labels(one.mlp) + [np.full(one.skip.shape, n_units)]
    l1, l2 = one.layers
    W = l1.weight.shape[0]
    return [
        np.broadcast_to(np.arange(W)[:, None], l1.weight.shape),  # w1 row
        np.arange(W),  # b1
        np.broadcast_to(np.arange(W)[None, :], l2.weight.shape),  # w2 col
        # With no hidden units (`nn_width=0`) there is nothing to spread over.
        np.arange(l2.bias.shape[0]) % max(W, 1),
    ]  # b2 spread


def _perm_from_labels(block, label_fn):
    arr, _ = eqx.partition(block, eqx.is_inexact_array)
    one = jax.tree.map(lambda l: l[0], arr)
    leaves = jax.tree.leaves(one)
    labels = label_fn(one)
    assert [tuple(l.shape) for l in leaves] == [
        tuple(np.shape(g)) for g in labels
    ], "label shapes must match jax.tree.leaves order of the block"
    lab = np.concatenate([np.asarray(g).ravel() for g in labels])
    return np.argsort(lab, kind="stable")


def make_plan(block, m, label_fn=None, q_min=16):
    """Sub-block plan for one bucket.  q ~ m/2, capped at Pb, so small
    conditioners stay unsplit (a single group covering the whole block)."""
    flatten, unflatten, Pb = codec(block)
    perm = (
        _perm_from_labels(block, label_fn)
        if (label_fn is not None and Pb > 0)
        else np.arange(Pb)
    )
    #q = int(min(Pb, max(q_min, m // 2))) if Pb else 1
    q = int(min(Pb, max(q_min, 2 * m))) if Pb else 1
    G = -(-Pb // q) if Pb else 1
    idx = np.full(G * q, Pb, dtype=int)
    idx[:Pb] = perm  # column Pb = dummy sink
    return dict(
        flatten=flatten,
        unflatten=unflatten,
        Pb=Pb,
        q=q,
        G=G,
        idx=jnp.asarray(idx.reshape(G, q)),
        mask=jnp.asarray((idx < Pb).reshape(G, q)).astype(float),
    )


def _gather(A, p):
    """(..., n, Pb) -> (..., n, G, q), padded entries zeroed."""
    A = jnp.concatenate([A, jnp.zeros(A.shape[:-1] + (1,), A.dtype)], -1)
    return A[..., p["idx"]] * p["mask"]


def _scatter(S, p):
    """(n, G, q) -> (n, Pb).  Live columns appear exactly once, so add == set."""
    n = S.shape[0]
    flat = jnp.zeros((n, p["Pb"] + 1), S.dtype).at[:, p["idx"]].add(S * p["mask"])
    return flat[:, : p["Pb"]]


# ============================================================ preconditioned CG


def pcg(Av, Minv, b, x0, rtol, maxiter):
    """PCG on pytrees.  Returns (x, n_iters).

    The stopping test is relative and taken in the norm the preconditioner
    induces, ``||r||_{M^-1}^2 = r . M^-1 r``: iterate until it falls below
    ``rtol`` times the same norm of ``b``. That is the norm PCG already
    monitors -- ``r . z`` is formed every iteration anyway, to build the next
    search direction -- so the test costs nothing, where a Euclidean ``||r||``
    costs a separate reduction over the whole parameter pytree per iteration.

    It is also the better-scaled test here. Under Marquardt damping the block
    curvatures span orders of magnitude, so a Euclidean residual weights each
    block by raw parameter scale; the ``M^-1`` norm weights it by that block's
    own curvature, which is what decides how much of the model decrease is
    still on the table.
    """
    tol_sq = rtol**2 * tdot(b, Minv(b))

    Ax0 = Av(x0)
    bx, xAx = tdot(b, x0), tdot(x0, Ax0)
    # Best multiple of the warm start: q(alpha*x0) = -(b.x0)^2 / (2 x0'A x0) <= 0,
    # so the monotone CG iterates keep q < 0 and the undamped pred > 0.
    # alpha = 0 (cold start) if x0 is uphill or zero.
    alpha = jnp.where((bx > 0) & (xAx > 0), bx / jnp.where(xAx > 0, xAx, 1.0), 0.0)
    x0 = jax.tree.map(lambda x: alpha * x, x0)
    r0 = jax.tree.map(lambda bb, ax: bb - alpha * ax, b, Ax0)
    z0 = Minv(r0)

    #r0 = jax.tree.map(lambda p, q: p - q, b, Av(x0))
    #z0 = Minv(r0)

    def cond(c):
        # `rz` is `r . z` for the carry's own `r`, so this tests the current
        # residual, not the previous one.
        _, _, _, _, rz, k = c
        return (rz > tol_sq) & (k < maxiter) & jnp.isfinite(rz)

    def body(c):
        x, r, p, _, rz, k = c
        Ap = Av(p)
        pAp = tdot(p, Ap)
        alpha = rz / jnp.where(pAp > 0, pAp, 1.0)
        x = jax.tree.map(lambda a, b_: alpha * a + b_, p, x)
        r = jax.tree.map(lambda a, b_: -alpha * a + b_, Ap, r)
        z = Minv(r)
        rz2 = tdot(r, z)
        p = jax.tree.map(lambda a, b_: (rz2 / rz) * a + b_, p, z)
        return x, r, p, z, rz2, k + 1

    x, r, _, _, _, k = jax.lax.while_loop(cond, body, (x0, r0, z0, z0, tdot(r0, z0), 0))
    return x, k


def capture_fractions(Gs, m):
    """Fraction of ``||J^T J||_F^2`` that each level of blocking captures.

    Answers whether the block-diagonal preconditioner is leaving anything on
    the table, and if so *where*, by splitting the mass three ways:

    * ``sub_block`` -- what `apply_Minvs` actually uses today, one block per
      (conditioner, sub-block).
    * ``conditioner`` -- what it would capture with ``G = 1``, i.e. if `m` were
      raised until `make_plan` stopped splitting conditioners. The gap to
      ``sub_block`` is exactly what removing the split would buy.
    * the remainder up to 1 is coupling *between* conditioners, which no
      block-diagonal preconditioner can see at any `m`; that is the part a
      low-rank/Nystrom correction would have to supply.

    The naive estimate is badly biased: ``Hhat`` is a Hutchinson average, so
    its off-diagonal mass contains estimator noise of order ``tr(H)^2 / m``,
    which at this parameter count swamps the signal and would make the block
    diagonal look far worse than it is. So the probes are split into
    independent halves and the cross term ``tr(Hhat_1 Hhat_2)`` is used, which
    is unbiased for ``||H||_F^2`` because the halves are independent.

    Everything is computed from the ``m x m`` cross-Gram of the two halves
    rather than from ``P x P`` blocks: ``tr(Hhat_1 Hhat_2) = ||A_1 A_2^T||_F^2
    / (m_1 m_2)``, and the ``1 / (m_1 m_2)`` cancels in the ratios. Summing
    that cross-Gram over sub-blocks gives the conditioner level and over
    everything gives the total, so one einsum per bucket yields all three.

    Both ratios are non-negative (each level is a trace of a product of PSD
    matrices) but are estimates, so they can exceed 1 by noise.
    """
    half = m // 2
    sub_mass = jnp.zeros(())
    conditioner_mass = jnp.zeros(())
    total_cross = jnp.zeros((half, half))
    for G in Gs:
        first, second = G[:half], G[half : 2 * half]
        # (n, G, m_1, m_2); the largest array here, so this is a diagnostic to
        # run occasionally rather than every step.
        cross = jnp.einsum("sngi,tngi->ngst", first, second)
        sub_mass = sub_mass + jnp.sum(cross**2)
        conditioner_mass = conditioner_mass + jnp.sum(jnp.sum(cross, axis=1) ** 2)
        total_cross = total_cross + jnp.sum(cross, axis=(0, 1))
    total = jnp.sum(total_cross**2)
    return {
        "sub_block": sub_mass / total,
        "conditioner": conditioner_mass / total,
    }


def build_blocks(vjpf, shape, key, plans, m, batch=32, capture=False):
    """[ (n_j, G_j, q_j, q_j) ] per bucket, estimated from m VJPs.

    E[(G^T w)(G^T w)^T] = G^T G for isotropic w, so a single VJP updates every
    block at once and the estimate is unbiased and PSD by construction.
    `vjpf` is the caller's already-linearized res_fn (see lm_step) -- forming
    it here too would make XLA build and optimize a second, identical copy of
    the residual function's forward+backward graph, which measurably bloats
    compile time.

    The Gram is accumulated chunk by chunk rather than by stacking every probe
    and contracting once at the end. `jax.lax.map`'s ``batch_size`` bounds how
    many probes are *computed* concurrently but still materializes all ``m`` of
    them, so the stacked probes cost ``m * P`` -- twice the blocks they are
    only an intermediate for, and the largest array in a step. Accumulating
    reduces that to ``batch * P`` transient, leaving the blocks themselves as
    the peak. Note this means ``batch`` now bounds peak memory properly;
    before, lowering it capped concurrency while the ``m * P`` stack remained.

    `capture` needs the individual probes, since `capture_fractions` cross-
    multiplies two independent halves, so it takes the stacking path and pays
    the ``m * P``. That is the diagnostic's price, and another reason to run it
    occasionally rather than on every step.
    """

    def one(k):
        w = jr.rademacher(k, shape).astype("float64")
        g = vjpf(w)[0]
        return [_gather(p["flatten"](b), p) for b, p in zip(blocks_of(g), plans)]

    keys = jr.split(key, m)
    totals, capture_info = _accumulate_grams(one, keys, batch, capture)
    return [total / m for total in totals], capture_info


def _accumulate_grams(one, items, batch, capture):
    """``sum_t G_t^T G_t`` per bucket, `G_t = one(items[t])` in gathered
    ``(n, G, q)`` form, taking `batch` items at a time. Returns ``(totals,
    capture_fractions or None)``; see `build_blocks` for why the two paths
    differ."""
    n_items = jax.tree.leaves(items)[0].shape[0]

    if capture:
        Gs = jax.lax.map(one, items, batch_size=batch)
        totals = [jnp.einsum("tngi,tngj->ngij", G, G) for G in Gs]
        return totals, capture_fractions(Gs, n_items)

    def chunk_gram(chunk):
        # An item may contribute several rows (leading axes before `(n, G, q)`),
        # as `build_blocks_exact`'s factors do; all of them are summed over.
        return [
            jnp.einsum(
                "tngi,tngj->ngij",
                G.reshape((-1,) + G.shape[-3:]),
                G.reshape((-1,) + G.shape[-3:]),
            )
            for G in jax.vmap(one)(chunk)
        ]

    # (n, G, q) per bucket -> (n, G, q, q) accumulators, from shapes alone.
    totals = [
        jnp.zeros(spec.shape[-3:] + spec.shape[-1:], spec.dtype)
        for spec in jax.eval_shape(one, jax.tree.map(lambda a: a[0], items))
    ]

    n_chunks, remainder = divmod(n_items, batch)
    if n_chunks:

        def accumulate(carry, chunk):
            return [c + g for c, g in zip(carry, chunk_gram(chunk))], None

        chunks = jax.tree.map(
            lambda a: a[: n_chunks * batch].reshape((n_chunks, batch) + a.shape[1:]),
            items,
        )
        totals, _ = jax.lax.scan(accumulate, totals, chunks)
    if remainder:
        tail = chunk_gram(jax.tree.map(lambda a: a[n_chunks * batch :], items))
        totals = [c + g for c, g in zip(totals, tail)]

    return totals, None


def build_blocks_exact(factor, data, plans, batch=32):
    """[ (n_j, G_j, q_j, q_j) ] per bucket: the exact Gauss-Newton blocks.

    `factor(draw_data)` returns one draw's factors, per bucket ``(n_j, r_j,
    Pb_j)`` with the block ``sum_draws V^T V`` (see
    `SparseTriangularMap.gauss_newton_factors`). No probes, so no estimator
    noise and no rank limit: the blocks are what `build_blocks` estimates, up
    to floating point. They are accumulated `batch` draws at a time, in the
    plans' sub-block layout, and scaled by ``1 / n_draws`` for `residuals`'
    ``1 / sqrt(n_draws)``.
    """
    n = jax.tree.leaves(data)[0].shape[0]

    def one(draw_data):
        return [
            _gather(jnp.swapaxes(V, 0, 1), p)  # (r, n_j, G, q)
            for V, p in zip(factor(draw_data), plans)
        ]

    totals, _ = _accumulate_grams(one, data, batch, capture=False)
    return [total / n for total in totals], None


def build_blocks_grouped(
    group_res_fn, data, theta, key, plans, n_groups, rounds, batch=32, capture=False
):
    """[ (n_j, G_j, q_j, q_j) ] per bucket, from per-group probes.

    `build_blocks` probes the residual of *every* draw at once, ``g = J^T w =
    sum_s J_s^T w_s``, so each sample's outer product carries all ``n^2`` draw
    pairs, and the ``s != t`` ones are zero-mean noise: entry variance grows
    like ``n^2 / m``, from ``m`` reverse passes over all draws. Here the draws
    are split into `n_groups` groups, each with its own probe, and the outer
    product is taken per group, so only pairs *within* a group contribute
    noise. Every group's gradient together costs one forward and reverse pass
    over the draws, so a round yields `n_groups` samples for the price of one
    of `build_blocks`' probes: variance ``~ n^2 / (n_groups * rounds)`` and rank
    up to ``n_groups * rounds``, from `rounds` passes. The estimate stays
    unbiased and PSD.

    The group gradients cannot come from `step`'s global `vjpf` -- a probe
    masked to one group still pays a full reverse pass -- so each group
    linearizes `group_res_fn` on its own draws, which adds up to one extra
    forward pass per round.

    `group_res_fn(theta, group_data)` must evaluate the residuals of the draws
    in `group_data` (leading axis: draws), normalized like `res_fn` by
    ``1 / sqrt(number of draws)``; the ``sqrt(group size / n)`` that turns
    group-normalized into globally normalized residuals is applied here.

    Groups are contiguous and equal-sized. When `n_groups` does not divide the
    draw count, the last group is padded by repeating draw 0, so its residual
    stays finite, and the padded draws get a zero probe, so they contribute
    nothing.

    `batch` is the number of groups taken at once, each carrying a forward and
    reverse pass over its own draws. `capture` needs two independent halves
    that each cover every draw, so it requires an even `rounds`.
    """
    n = jax.tree.leaves(data)[0].shape[0]
    n_groups = min(n_groups, n)
    size = -(-n // n_groups)
    flat = np.arange(n_groups * size)
    index = np.where(flat < n, flat, 0).reshape(n_groups, size)
    mask = jnp.asarray((flat < n).reshape(n_groups, size), dtype=float)
    grouped = jax.tree.map(lambda a: a[index], data)
    if capture and rounds % 2:
        raise ValueError("capture_diagnostic with grouped probes needs even rounds.")

    def one(item):
        group, k = item
        group_data = jax.tree.map(lambda a: a[group], grouped)
        out, vjpg = jax.vjp(lambda p: group_res_fn(p, group_data), theta)
        group_mask = mask[group].reshape((size,) + (1,) * (out.ndim - 1))
        w = jr.rademacher(k, out.shape, dtype=out.dtype) * group_mask
        g = vjpg(w)[0]
        return [_gather(p["flatten"](b), p) for b, p in zip(blocks_of(g), plans)]

    # Round-major, so each half of the items (for `capture`) is whole rounds.
    groups = jnp.asarray(np.tile(np.arange(n_groups), rounds))
    keys = jr.split(key, n_groups * rounds)
    totals, capture_info = _accumulate_grams(one, (groups, keys), batch, capture)
    return [total * (size / n) / rounds for total in totals], capture_info


def block_diagonal(blocks, plans, template, floors):
    """`diag(J^T J)`, floored per conditioner at `floors`, as a pytree shaped
    like `template`.

    Free: `build_blocks` already estimates each block's Gram matrix, and its
    diagonal is exactly the corresponding run of `diag(J^T J)` -- a plan's
    column index is a permutation, so every live parameter sits in exactly one
    sub-block, in exactly one position.
    """
    out = [
        p["unflatten"](
            _scatter(
                jnp.maximum(
                    jnp.diagonal(H, axis1=-2, axis2=-1), floor[:, None, None]
                ),
                p,
            )
        )
        for p, H, floor in zip(plans, blocks, floors)
    ]
    # `unflatten` recombines each block's static leaves, which `template` (a
    # filtered params pytree) does not carry; drop them again so the result can
    # be tree_map'd against tangents leaf-for-leaf.
    return eqx.filter(rebuild(template, out), eqx.is_inexact_array)


# Fraction of a conditioner's largest curvature below which Marquardt damping
# is floored. Must not be tiny: `zero_init` shrinks each conditioner's last
# layer to near zero, so the first-layer parameters start with (near-)zero
# gradient and curvature, and an unfloored `lam * diag(J^T J)` leaves them
# undamped -- both in CG and, far worse, in the block preconditioner, whose
# inverse then carries entries of order `1 / (lam * floor)`.
#
# Measured with a *global* floor on the 10-dim funnel (arrow pattern, 10 LM
# steps, final F / steps accepted): 1e-8 -> 1.6e+02, 1/10 (diverges); 1e-4 ->
# 1.4e-02, 8/10; 1e-2 -> 4.8e-02, 9/10; 1e-1 -> 8.9e-02, 10/10. Too low and the
# flat directions blow up; too high and this degrades towards absolute
# damping, which is what it exists to avoid.
MARQUARDT_FLOOR = 1e-4
# Fraction of the largest curvature *anywhere*, as a fallback for conditioners
# that are flat as a whole, which a purely local floor would leave undamped.
MARQUARDT_GLOBAL_FLOOR = 1e-10


def marquardt_floors(blocks):
    """Smallest damping scale allowed, one per conditioner: ``(n,)`` per bucket.

    Marquardt damping is `lam * diag(J^T J)`, which vanishes wherever the
    curvature does. The floor is relative to each conditioner's own largest
    curvature rather than the global one, so that a conditioner whose
    curvature sits orders of magnitude below the stiffest one is still damped
    by its own scale instead of being flattened towards absolute damping.
    """
    local = [
        jnp.max(jnp.diagonal(H, axis1=-2, axis2=-1), axis=(1, 2)) for H in blocks
    ]
    global_max = jnp.maximum(jnp.max(jnp.concatenate(local)), 1e-300)
    return [
        jnp.maximum(MARQUARDT_FLOOR * dmax, MARQUARDT_GLOBAL_FLOOR * global_max)
        for dmax in local
    ]


def _block_inv(H, lam, floor, shrinkage):
    """Inverse of the damped block, via Cholesky.

    `floor is None` selects absolute (Levenberg) damping, `lam * I`; otherwise
    damping is Marquardt's `lam * diag(H)`, floored at `floor`. With one block
    per conditioner, an absolute `lam` is a single trust region shared by
    thousands of blocks whose curvature scales have no reason to agree -- the
    conditioners and the affine tail least of all -- so it ends up set by
    whichever block wants the smallest step, and every other block takes a step
    far shorter than it could. Scaling by each block's own diagonal makes `lam`
    dimensionless per block. Which is better is problem-dependent, hence
    `step`'s `damping` argument; whichever is chosen, `floor` must match the
    one `step` applies in `Av`, or the preconditioner would approximate a
    different operator than CG is solving.

    `H` is first shrunk towards its diagonal, ``(1 - s) H + s diag(H)``. It is
    a Hutchinson estimate from `m` probes, so it has rank at most `m`, and its
    small eigen-directions are unreliable well before that unless ``m >> q``.
    Unshrunk, directions the probes missed fall back to `lam * D` alone, and as
    `lam` decays the preconditioned operator there grows like ``1 / lam``;
    shrunk, they fall back to Jacobi. The diagonal is unchanged, so `D` stays
    consistent with the `Av` operator.

    `H` is a Gram matrix (see `build_blocks`), hence PSD, so `H + lam D` with
    `D` a positive diagonal is positive definite; the extra ridge keeps that
    true *numerically* once `lam` has decayed towards `lam_min`. Since this is
    only a preconditioner, perturbing it costs CG iterations, never
    correctness.

    Cholesky rather than `eigh` because cuSOLVER's batched `syev` allocates a
    device workspace proportional to the entire batch, and the batch here is
    one block per conditioner per sub-block -- so it grows with the model while
    `q` stays fixed, and it is the first thing to exhaust device memory on a
    large model. Cholesky needs no such workspace.
    """
    if False:
        # eigh with clamping rather than Cholesky: blocks are small, some are
        # structurally singular (a conditioner with in_features=0 has a
        # draw-independent Jacobian), and eigh avoids the static `lower` flag
        # that breaks under vmap.
        w, V = jnp.linalg.eigh(H)
        return jnp.einsum("ij,j,kj->ik", V, 1.0 / (jnp.maximum(w, 0.0) + lam), V)

    q = H.shape[-1]
    eye = jnp.eye(q, dtype=H.dtype)
    H = (1.0 - shrinkage) * H + shrinkage * jnp.diag(jnp.diagonal(H))
    if floor is None:
        damped = H + (lam + 1e-12 * jnp.maximum(jnp.mean(jnp.diagonal(H)), 1.0)) * eye
    else:
        diag = lam * jnp.maximum(jnp.diagonal(H), floor) + 1e-4 * floor
        damped = H + jnp.diag(diag)
    factor = jnp.linalg.cholesky(damped)
    inv_factor = solve_triangular(factor, eye, lower=True)
    return inv_factor.T @ inv_factor


def precompute_Minvs(blocks, plans, n_samples, lam, floors):
    """(n, G, q, q) per bucket.  Computed once per lm_step call and reused
    across every Minv(v) call inside that step's PCG loop.

    `floors` is `marquardt_floors`' output, or `None` for absolute damping,
    which then stays a plain trace-time branch inside `_block_inv` instead of
    something vmap has to map over.

    The shrinkage weight ``q / (q + n_samples)`` is the usual covariance-
    shrinkage scale: negligible once the samples behind the estimate (probes,
    or groups times rounds) outnumber the block's size, about half when they
    match it.
    """
    out = []
    for i, (H, p) in enumerate(zip(blocks, plans)):
        shrinkage = p["q"] / (p["q"] + n_samples)
        if floors is None:
            inv = lambda H: _block_inv(H, lam, None, shrinkage)
            out.append(jax.vmap(jax.vmap(inv))(H))
        else:
            inv = lambda H, floor: _block_inv(H, lam, floor, shrinkage)
            out.append(jax.vmap(jax.vmap(inv, in_axes=(0, None)))(H, floors[i]))
    return out


def apply_Minvs(Minvs, plans, v):
    out = []
    for b, p, Mi in zip(blocks_of(v), plans, Minvs):
        S = _gather(p["flatten"](b), p)  # (n, G, q)
        S = jnp.einsum("ngij,ngj->ngi", Mi, S)
        out.append(p["unflatten"](_scatter(S, p)))
    return rebuild(v, out)


def blocks_zeros(theta, plans):
    """Zero accumulators shaped exactly like `build_blocks`' output.

    Only needed so that `blocks` has a stable pytree structure across `step`
    calls: handing it `None` on the first step and a list of arrays afterwards
    means the second call sees a different argument structure and retraces,
    recompiling the whole step -- CG loop, probe map and the residual function's
    forward and backward graph with it.

    The shapes come from `_gather` alone, mirroring how `build_blocks` derives
    its own accumulators, and the VJP that produces the real values has the same
    structure as `theta`, so `theta` stands in for it here. That keeps this off
    `res_fn` entirely: no VJP is traced and nothing is compiled. Should the two
    ever disagree, `step`'s `lax.cond` fails loudly rather than silently, since
    its branches must return matching shapes.
    """
    spec = jax.eval_shape(
        lambda t: [_gather(p["flatten"](b), p) for b, p in zip(blocks_of(t), plans)],
        theta,
    )
    return [jnp.zeros(s.shape + s.shape[-1:], s.dtype) for s in spec]


@eqx.filter_jit
def step(
    res_fn,
    args,
    data,
    plans,
    theta,
    r,
    lam,
    key,
    *,
    frozen,
    rebuild_blocks: jax.Array,
    p_prev: jax.Array,
    eta_in: jax.Array,
    nu_in: jax.Array,
    m=256,
    batch=32,
    precondition=True,
    cg_tol=1e-2,
    cg_eta_max=0.5,
    cg_gamma=0.9,
    cg_alpha=1.618,
    cg_max=300,
    accept_rho=0.1,
    nu0=2.0,
    lam_min=None,
    lam_max=1e10,
    damping="marquardt",
    capture_diagnostic=False,
    blocks=None,
    n_groups=None,
    rounds=1,
    line_search=False,
    ls_min_fraction=0.1,
    forcing="residual",
    factor_fn=None,
):
    """One LM step.  res_fn and plans are static under eqx.filter_jit; plans
    is built once via get_plans and cached by parameter structure, so its
    identity (and that of the flatten/unflatten closures it holds) stays
    stable across steps and across repeated `fit` calls with the same
    architecture.

    `blocks` carries the Gram estimate in from the previous step, reused when
    `rebuild_blocks` is false. Estimating it costs `m` reverse passes, several
    times a whole solve once CG is cheap, so reuse is the dominant cost lever;
    `fit` decides the policy. Only `Hb` is carried -- `Minvs` depends on `lam`
    and is rebuilt every step, but that is small Choleskys with no VJPs.

    `rebuild_blocks` is a *traced* predicate selecting between the two through
    `lax.cond`, not a static flag: `fit` flips it on a schedule, and a static
    flag would mean carrying two compiled variants of this function. The
    branches differ by an `m`-probe `lax.map`, far too large for XLA to
    predicate, so the untaken one really is skipped.

    `eta_in` is the CG forcing term this step should use, proposed by the
    previous step (see the Eisenstat-Walker comment below); the successor is
    returned as ``info["cg_eta_next"]`` for `fit` to feed back. It is traced,
    like `rebuild_blocks`. `nu_in` is the companion state for the `lam` update,
    returned as ``info["lam_nu_next"]``.

    `theta` is the trainable part only; `frozen` is merged back in for every
    residual evaluation, see `split_frozen`.

    `n_groups` selects the block estimator: `None` probes all draws at once
    with `m` probes (`build_blocks`), an integer uses `rounds` rounds of
    per-group probes over `data` (`build_blocks_grouped`). Either way `m` sets
    the sub-block size through `plans`.
    """
    group_res_fn = lambda p, d: res_fn(merge_frozen(p, frozen), (*args, *d))
    res_fn_args = lambda p: group_res_fn(p, data)
    _, vjpf = jax.vjp(res_fn_args, theta)
    vjp = lambda w: vjpf(w)[0]

    if damping not in ("marquardt", "absolute"):
        raise ValueError(
            f"Unknown damping {damping!r}, expected 'marquardt' or 'absolute'."
        )
    if forcing not in ("residual", "rho"):
        raise ValueError(f"Unknown forcing {forcing!r}, expected 'residual' or 'rho'.")
    if lam_min is None:
        # Under Marquardt damping `lam` is dimensionless -- a fraction of each
        # block's own curvature -- so the 1e-10 that made sense for absolute
        # damping means "undamped to well below double precision", and CG ends
        # up solving a near-singular system to a tolerance it cannot reach.
        lam_min = 1e-6 if damping == "marquardt" else 1e-10

    if precondition:
        if factor_fn is not None:
            if capture_diagnostic:
                raise ValueError(
                    "capture_diagnostic needs probe-based blocks, not factor_fn."
                )
            # Exact, so nothing to shrink towards the diagonal.
            n_samples = float("inf")
            estimate = lambda capture: build_blocks_exact(
                lambda d: factor_fn(merge_frozen(theta, frozen), args, d),
                data,
                plans,
                batch,
            )
        elif n_groups is None:
            n_samples = m
            estimate = lambda capture: build_blocks(
                vjpf, r.shape, key, plans, m, batch, capture=capture
            )
        else:
            n_samples = n_groups * rounds
            estimate = lambda capture: build_blocks_grouped(
                group_res_fn, data, theta, key, plans, n_groups, rounds, batch,
                capture=capture,
            )
        if blocks is None or capture_diagnostic:
            # Nothing to reuse on the first step, and the diagnostic needs the
            # individual probes, so both always build.
            Hb, capture = estimate(capture_diagnostic)
        else:
            capture = None
            Hb = jax.lax.cond(
                rebuild_blocks, lambda: estimate(False)[0], lambda: blocks
            )
        if damping == "marquardt":
            floors = marquardt_floors(Hb)
            D = block_diagonal(Hb, plans, theta, floors)
        else:
            D = floors = None
        Minvs = precompute_Minvs(Hb, plans, n_samples, lam, floors)
        Minv = lambda v: apply_Minvs(Minvs, plans, v)
    else:
        # Without the block estimates there is no per-parameter curvature to
        # scale by, so Marquardt damping is not available here.
        Hb = D = capture = None
        Minv = lambda v: v

    jvp = lambda v: jax.jvp(res_fn_args, (theta,), (v,))[1]
    if D is None:
        Av = lambda v: jax.tree.map(lambda a, b: a + lam * b, vjp(jvp(v)), v)
    else:
        Av = lambda v: jax.tree.map(
            lambda a, d, b: a + lam * d * b, vjp(jvp(v)), D, v
        )

    g = vjp(r)
    rhs = jax.tree.map(jnp.negative, g)
    x0 = p_prev

    # `cg_tol` is the floor on the forcing term and `cg_eta_max` the cap;
    # setting them equal recovers fixed-tolerance behaviour exactly. The value
    # itself was proposed by the previous step, see `eta_next` below.
    eta = jnp.clip(eta_in, cg_tol, cg_eta_max)

    p, ncg = pcg(Av, Minv, rhs, x0, eta, cg_max)

    theta_new = jax.tree.map(jnp.add, theta, p)
    r_new = res_fn_args(theta_new)
    Jp = jvp(p)

    F, F_new = tdot(r, r), tdot(r_new, r_new)
    step_length = jnp.ones((), F.dtype)
    # Quality of the full GN step, which is what `lam` controls. The line
    # search below only picks how much of that step to take; were `lam` fed
    # the shortened step's `rho` instead, a well-predicted short step would
    # keep lowering `lam`, lengthening the GN step it is cut from, until the
    # search sits at its minimum fraction along an ever longer direction.
    pred_full = -tdot(p, g) - 0.5 * tdot(Jp, Jp)
    rho_full = 0.5 * (F - F_new) / jnp.where(
        jnp.abs(pred_full) < 1e-30, 1e-30, pred_full
    )
    full_step_good = (
        (rho_full > accept_rho)
        & (pred_full > 0)
        & jnp.isfinite(F_new)
        & jnp.isfinite(rho_full)
    )

    if line_search:
        # On a large-residual problem the dropped second-order term makes the
        # GN model underestimate curvature -- by about 2x along scale
        # directions -- so every step overshoots and the error flips sign
        # rather than shrinking; damping only slows that into a creep, and
        # Nielsen's update is stuck there, since its fixed point is exactly
        # `rho = 0.5`. A parabola through f(0), f'(0) = g.p and f(1), with
        # f = F / 2 along `p`, measures the curvature actually met and picks
        # the step fraction that minimizes it: about 1/2 in that regime. It
        # costs one residual evaluation, and only when it shortens the step.
        slope = tdot(p, g)
        curvature = 0.5 * (F_new - F) - slope
        fraction = -slope / (2.0 * jnp.where(curvature > 0, curvature, 1.0))
        fraction = jnp.clip(fraction, ls_min_fraction, 1.0)
        shorten = jnp.isfinite(F_new) & (curvature > 0) & (fraction < 0.95)
        p_short = jax.tree.map(lambda a: fraction * a, p)
        r_short = jax.lax.cond(
            shorten,
            lambda: res_fn_args(jax.tree.map(jnp.add, theta, p_short)),
            lambda: r_new,
        )
        F_short = tdot(r_short, r_short)
        # Kept only if it actually beats the full step; a NaN compares False.
        take = shorten & (F_short < F_new)
        pick_short = lambda a, b: jax.tree.map(lambda x, y: jnp.where(take, x, y), a, b)
        p, r_new = pick_short(p_short, p), pick_short(r_short, r_new)
        # `J` is linear, so the shortened step's `Jp` is just rescaled.
        Jp = jax.tree.map(lambda a: jnp.where(take, fraction, 1.0) * a, Jp)
        theta_new = jax.tree.map(jnp.add, theta, p)
        F_new = jnp.where(take, F_short, F_new)
        step_length = jnp.where(take, fraction, 1.0)

    actual = 0.5 * (F - F_new)
    pred = -tdot(p, g) - 0.5 * tdot(Jp, Jp)  # undamped GN model
    rho = actual / jnp.where(jnp.abs(pred) < 1e-30, 1e-30, pred)

    ok = jnp.isfinite(F_new) & jnp.isfinite(rho)
    # `pred > 0` is not implied by `rho > accept_rho`: `pred < 0` means `p` is
    # not a descent direction, `actual` is then negative too, and the signs
    # cancel into a large positive `rho`.
    accept = (rho > accept_rho) & (pred > 0) & ok

    pick = lambda a, b: jax.tree.map(lambda x, y: jnp.where(accept, x, y), a, b)
    theta_out, r_out = pick(theta_new, theta), pick(r_new, r)

    # Nielsen's damping update (Madsen/Nielsen/Tingleff), in place of a fixed
    # decrease/increase pair. Two properties matter here.
    #
    # The decrease is graded by `rho` and capped at 3x, so a step that only
    # just cleared `accept_rho` *raises* `lam` (the factor exceeds 1 below
    # `rho = 0.5`) instead of leaving it put -- a fixed `lam / lam_down` on
    # every good step keeps proposing a trust region the problem has already
    # refused.
    #
    # The increase escalates: `nu` doubles on each consecutive failure and
    # resets on acceptance. A fixed multiplier produces a limit cycle here --
    # accept at `rho ~ 0.9`, divide `lam`, get `rho < 0` at the smaller value,
    # multiply back to almost exactly the `lam` that just worked, repeat -- and
    # each of those probes costs a full CG solve plus a residual and Jacobian
    # evaluation. Escalating means a second consecutive failure leaves the
    # interval instead of retracing it.
    lam_decrease = jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho_full - 1.0) ** 3)
    lam_next = jnp.where(full_step_good, lam * lam_decrease, lam * nu_in)
    nu_out = jnp.where(full_step_good, nu0, 2.0 * nu_in)
    # When the line search took (and kept) only a fraction `a` of the GN step,
    # the step `lam` produced was ~1/a too long. Where damping dominates the
    # curvature, step length goes like 1 / lam, so `lam / a` sizes the next GN
    # step about right in one go -- rather than letting Nielsen's rules halve
    # or triple their way there while CG keeps solving tightly for steps that
    # are mostly thrown away. Where the curvature dominates it undercorrects,
    # and the usual rules take over on the following steps.
    shortened = accept & (step_length < 1.0)
    lam_next = jnp.where(shortened, lam / step_length, lam_next)
    nu_out = jnp.where(shortened, nu0, nu_out)
    lam_out = jnp.clip(lam_next, lam_min, lam_max)
    # a rejected step's p solves a system with a different lambda: discard it
    p_out = jax.tree.map(lambda a: jnp.where(accept, a, jnp.zeros_like(a)), p)

    # Inexact-Newton forcing term for the *next* step, Eisenstat-Walker choice
    # 1: how badly the linear model mispredicted the residual the step actually
    # produced,
    #
    #     eta_next = | ||r_new|| - ||r + J p|| | / ||r||.
    #
    # Solving the linear system more accurately than the linear model deserves
    # buys nothing, and this measures that directly, from quantities the step
    # already formed. It needs no assumption that anything decreases
    # monotonically -- which rules out the residual-ratio rules (EW choice 2,
    # and the textbook `min(eta_max, sqrt(||g||))`): `||g|| = ||J^T r||` rises
    # on a good fraction of accepted steps here, and a ratio rule reacts to a
    # rising gradient by demanding its *loosest* solve, exactly when the
    # iteration is doing worst. It is also scale invariant.
    #
    # It does not adapt on a large-residual problem, though: where `F`
    # plateaus well above zero, both norms in the numerator are ~||r|| and
    # differ only at second order in the step, so the ratio collapses to
    # `cg_tol` exactly when the solves stop mattering. `forcing="rho"` instead
    # measures the model error against the *predicted decrease*,
    #
    #     eta_next = |1 - rho|,
    #
    # with `rho` that of the full GN step (the one CG solved for, whatever the
    # line search then takes). A model that is only roughly right on the
    # plateau, `rho ~ 0.5-0.9`, then gets a correspondingly rough solve.
    if forcing == "rho":
        eta_next = jnp.abs(1.0 - rho_full)
        # A full step that blew up has no finite `rho`, but the line search
        # may still have accepted a shortened one; the model is then as poor
        # as it gets, so ask for the loosest solve rather than pass on a NaN,
        # which `jnp.clip` would keep.
        eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, cg_eta_max)
    else:
        linear_r_norm = tnorm(jax.tree.map(jnp.add, r, Jp))
        eta_next = jnp.abs(jnp.sqrt(F_new) - linear_r_norm) / jnp.sqrt(
            jnp.where(F > 0, F, 1.0)
        )
    # EW's safeguard against an over-rapid decrease: without it a single
    # unusually accurate model demands a near-exact solve on the very next
    # step, which is the `cg_max`-burning failure mode the forcing term exists
    # to avoid. Skipped where it would ask for a loose solve anyway.
    safeguard = cg_gamma * eta**cg_alpha
    eta_next = jnp.where(safeguard > 0.1, jnp.maximum(eta_next, safeguard), eta_next)
    # A rejected step's `p` is thrown away, so the model quality measured along
    # it says nothing about the step that will be taken instead: hold `eta`.
    eta_next = jnp.where(accept & ok, eta_next, eta)

    info = {
        "F": F,
        "F_new": F_new,
        # Loss of the state actually carried forward: `F_new` is the proposal,
        # which a rejected step throws away.
        "F_out": jnp.where(accept, F_new, F),
        "accept": accept,
        "rho": rho,
        "actual": actual,
        "pred": pred,
        "lam_in": lam,
        "lam_out": lam_out,
        "lam_nu": nu_in,
        "lam_nu_next": nu_out,
        "n_cg": ncg,
        "cg_eta": eta,
        "cg_eta_next": eta_next,
        "cg_converged": ncg < cg_max,
        "grad_norm": tnorm(g),
        "step_norm": tnorm(p),
        "finite": ok,
        "rebuilt_blocks": rebuild_blocks,
        "step_length": step_length,
    }
    if capture is not None:
        info["capture"] = capture
    # Diagnostics only -- `fit` never prints these -- but a second full batched
    # eigendecomposition per step, hitting the same cuSOLVER workspace limit as
    # `_block_inv` used to. Off by default; flip to re-enable.
    if False and plans is not None:
        info["block_min_eig"] = jnp.stack([jnp.min(jnp.linalg.eigvalsh(H)) for H in Hb])
        info["block_max_eig"] = jnp.stack([jnp.max(jnp.linalg.eigvalsh(H)) for H in Hb])
    return theta_out, r_out, lam_out, p_out, Hb, info


# ============================================================ setup and driver

_plans_cache = {}


def _plans_signature(params, m, q_min):
    """Hashable structural fingerprint of params (treedef + leaf shapes/
    dtypes) plus the plan hyperparameters.  Used to key _plans_cache so that
    two calls with the same architecture reuse the same plans object instead
    of building fresh (and differently-identified) flatten/unflatten
    closures each time."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = tuple((tuple(np.shape(l)), np.result_type(l).str) for l in leaves)
    return (treedef, shapes, m, q_min)


def get_plans(params, m=128, q_min=16):
    """Block plans for this parameter structure, built once and cached.

    Plans hold flatten/unflatten closures (from ravel_pytree) that end up as
    static arguments to eqx.filter_jit(lm_step); without caching, every call
    would create new closures with a new identity and force a recompile.
    """
    key = _plans_signature(params, m, q_min)
    if key not in _plans_cache:
        n_conditioners = len(_conditioners(params))
        _plans_cache[key] = [
            # The affine tail has no hidden units to group by.
            make_plan(b, m, mlp_unit_labels if i < n_conditioners else None, q_min)
            for i, b in enumerate(blocks_of(params))
        ]
    return _plans_cache[key]


def setup(params, m=128, q_min=16):
    """Return the block plans lm_step expects."""
    plans = get_plans(params, m, q_min)

    assert jax.tree.all(
        jax.tree.map(
            lambda a, b: bool(jnp.all(a == b)),
            rebuild(params, blocks_of(params)),
            params,
        )
    ), "blocks_of / rebuild do not round-trip"

    return plans


def fit(
    params,
    res_fn,
    args,
    m=64,
    n_steps=60,
    lam0=1e-2,
    seed=0,
    verbose=True,
    precondition=True,
    min_loss=None,
    cg_max=300,
    cg_eta_max=0.5,
    cg_gamma=0.9,
    cg_alpha=1.618,
    batch=32,
    damping="marquardt",
    lam_min=None,
    nu0=2.0,
    capture_diagnostic=False,
    rebuild_every=1,
    data=(),
    n_groups=None,
    rounds=1,
    fit_affine=True,
    rtol=None,
    patience=5,
    line_search=False,
    forcing="residual",
    factor_fn=None,
):
    """Levenberg-Marquardt fit.

    `factor_fn(params, args, draw_data)`, if given, supplies one draw's exact
    Gauss-Newton block factors (see `build_blocks_exact`), replacing the
    probe estimates: `m` then only sets the sub-block size, and nothing is
    shrunk. It covers the conditioners only, so it needs `fit_affine=False`.

    `line_search` shortens each step to the minimizer of a parabola fitted
    along it, which counters the Gauss-Newton overshoot on large-residual
    problems (see `step`); the fraction taken shows as ``a=`` in the log.
    A shortened step also raises `lam` by ``1 / a``, so the next GN step comes
    out about the length the line search found.

    `forcing` picks how the CG tolerance adapts, ``"residual"``
    (Eisenstat-Walker choice 1) or ``"rho"`` (``|1 - rho|``, which keeps
    adapting when the loss plateaus well above zero); see `step`.

    Stops after `n_steps`, once the loss falls below `min_loss`, or -- if
    `rtol` is given -- once `patience` consecutive steps have together lowered
    the loss by less than a fraction `rtol` of it. Rejected steps count
    towards `patience`: a run of rejections at rising `lam` is stagnation too.

    Residuals are ``res_fn(params, (*args, *data))``. `data` holds the
    per-draw arrays (leading axis: draws), kept apart from `args` so that the
    grouped block estimator can evaluate subsets of draws, see
    `build_blocks_grouped` for what `res_fn` must then satisfy. `n_groups`
    selects that estimator, with `rounds` rounds of probes; `None` keeps
    `build_blocks`' `m` probes over all draws. `m` sets the sub-block size
    either way.

    `damping` selects the trust region: ``"marquardt"`` damps with
    ``lam * diag(J^T J)``, ``"absolute"`` with the classical ``lam * I`` (see
    `_block_inv`). Note that `lam0` and the `lam_min`/`lam_max` bounds mean
    different things under the two -- under ``"marquardt"`` `lam` is
    dimensionless, scaled by each block's own curvature -- so `lam` traces are
    not comparable across the choice and `lam0` may want retuning.

    `batch` is `build_blocks`' probe concurrency: how many of the `m` Rademacher
    VJPs are taken at once. It is the main memory knob of a step, because each
    concurrent probe carries a full reverse pass through the residual function,
    and that cost multiplies with whatever batching the residual function does
    internally. Lowering it trades sequential chunks (`m / batch`) for peak
    memory at no extra FLOPs.

    The diagonal affine tail is held fixed at its value in `params` unless
    `fit_affine`, in which case it gets preconditioner blocks of its own, one
    per variable; see `split_frozen`.
    """
    params, frozen = split_frozen(params, fit_affine)
    data = tuple(data)
    if n_groups is not None and not data:
        raise ValueError("n_groups needs the per-draw arrays passed as `data`.")
    if factor_fn is not None:
        if not data:
            raise ValueError("factor_fn needs the per-draw arrays passed as `data`.")
        if fit_affine:
            raise ValueError(
                "factor_fn only gives blocks for the conditioners; use "
                "fit_affine=False with it."
            )

    if precondition:
        plans = setup(params, m)

        if verbose:
            print(
                "bucket (Pb, q, n_subblocks):",
                [(p["Pb"], p["q"], p["G"]) for p in plans],
            )
    else:
        plans = None

    key = jax.random.key(0)

    theta, r = params, res_fn(merge_frozen(params, frozen), (*args, *data))
    # Seed with real zeros (not None) so p_prev's pytree structure is the
    # same on every iteration -- else step 0 (p_prev=None) and step 1
    # (p_prev=<pytree>) are different input structures and each triggers its
    # own trace of lm_step (and the pcg while_loop inside it).
    p_prev = jax.tree.map(jnp.zeros_like, params)
    # Every scalar carried into `step` is seeded at `r`'s dtype rather than left
    # to `jnp.asarray(<python float>)`, which produces a *weakly* typed array.
    # What `step` returns is arithmetic on `r` and so is strongly typed, and a
    # weak -> strong flip is a different input aval: it retraces and recompiles
    # the whole step, CG loop included, on the second call.
    #
    # Forcing term (see `step`): each step proposes the next one's, so all this
    # carries is the seed. `cg_eta_max` is the usual Eisenstat-Walker `eta_0` --
    # there is no model-quality measurement yet to derive one from. `nu` is the
    # companion seed for Nielsen's `lam` update.
    best_loss, stalled = float(tdot(r, r)), 0
    lam = jnp.asarray(lam0, dtype=r.dtype)
    eta_prev = jnp.asarray(cg_eta_max, dtype=r.dtype)
    nu_prev = jnp.asarray(nu0, dtype=r.dtype)
    key, hist = jr.key(seed), []
    # The Gram estimate is a function of `theta` alone, so it stays exactly
    # valid across a rejected step -- `theta` did not move. Between accepted
    # steps it only drifts, so `rebuild_every` trades a slightly stale
    # preconditioner (which costs CG iterations, never correctness) against `m`
    # reverse passes. Counting only accepted steps makes the reuse on rejection
    # exact rather than an approximation.
    # Placeholder with the right structure rather than `None`, see
    # `blocks_zeros`. The values are never read: `rebuild_blocks` is true on the
    # first step. Without preconditioning `step` ignores the argument entirely,
    # and `None` is then stable across calls.
    blocks = blocks_zeros(theta, plans) if precondition else None
    accepted_since_build = rebuild_every  # build on the first step
    for i in range(n_steps):
        key, sk = jr.split(key)
        rebuild_blocks = accepted_since_build >= rebuild_every
        theta, r, lam, p_prev, blocks, info = step(
            res_fn,
            args,
            data,
            plans,
            theta,
            r,
            lam,
            sk,
            frozen=frozen,
            m=m,
            precondition=precondition,
            p_prev=p_prev,
            eta_in=eta_prev,
            nu_in=nu_prev,
            nu0=nu0,
            cg_max=cg_max,
            cg_eta_max=cg_eta_max,
            cg_gamma=cg_gamma,
            cg_alpha=cg_alpha,
            batch=batch,
            damping=damping,
            lam_min=lam_min,
            capture_diagnostic=capture_diagnostic,
            blocks=blocks,
            rebuild_blocks=jnp.asarray(rebuild_blocks),
            n_groups=n_groups,
            rounds=rounds,
            line_search=line_search,
            forcing=forcing,
            factor_fn=factor_fn,
        )
        if rebuild_blocks:
            accepted_since_build = 0
        accepted_since_build += int(info["accept"])
        eta_prev = info["cg_eta_next"]
        nu_prev = info["lam_nu_next"]
        hist.append(info)
        if verbose:
            print(
                f"{i:3d}  F={float(info['F_new']):.4e}  "
                f"log(F)={float(np.log(info['F_new'])):+.2f} "
                f"rho={float(info['rho']):+.2f}  "
                f"lam={float(info['lam_out']):.1e}  "
                f"cg={int(info['n_cg']):3d}{' ' if info['cg_converged'] else '*'} "
                f"eta={float(info['cg_eta']):.2f}"
                f"{' ' if info['rebuilt_blocks'] else '~'}  "
                f"|g|={float(info['grad_norm']):.2e}"
                + (f"  a={float(info['step_length']):.2f}" if line_search else "")
                + (
                    "  capture sub/cond="
                    f"{float(info['capture']['sub_block']):.2f}/"
                    f"{float(info['capture']['conditioner']):.2f}"
                    if "capture" in info
                    else ""
                )
                + f"{'' if info['accept'] else '   REJECT'}"
            )

        if not np.isfinite(info["grad_norm"]):
            if verbose:
                print("gradient norm is NaN or Inf; adding noise to parameters")
            # Jitter `theta` rather than overwrite it: the iterate itself is
            # usually fine and only the local geometry is degenerate, so
            # discarding the fit so far costs every step taken to here.
            # One key per leaf -- reusing `sk` gives equal-shaped leaves
            # identical noise -- and only inexact leaves are perturbed.
            leaves, treedef = jax.tree.flatten(theta)
            keys = jr.split(sk, len(leaves))
            theta = jax.tree.unflatten(
                treedef,
                [
                    leaf + 1e-2 * jr.normal(k, jnp.shape(leaf), jnp.result_type(leaf))
                    if eqx.is_inexact_array(leaf)
                    else leaf
                    for leaf, k in zip(leaves, keys)
                ],
            )
            # Everything carried forward describes the *old* `theta`: `r` would
            # make the next step compute `J(theta_new)^T r_old` and an `F` from
            # a point it has left, `p_prev` warm-starts CG from a step of a
            # system that no longer exists, and the Gram blocks were estimated
            # elsewhere. Recompute or discard all three.
            r = res_fn(merge_frozen(theta, frozen), (*args, *data))
            p_prev = jax.tree.map(jnp.zeros_like, theta)
            accepted_since_build = rebuild_every  # rebuild at the new theta
            # The model-quality measurement the forcing term is built on refers
            # to the old point too; restart it, and the `lam` escalation with
            # it, from their seeds -- at `r`'s dtype, as above, so the restart
            # does not recompile `step`.
            eta_prev = jnp.asarray(cg_eta_max, dtype=r.dtype)
            nu_prev = jnp.asarray(nu0, dtype=r.dtype)
            # A new point, so progress is measured afresh from its loss.
            best_loss, stalled = float(tdot(r, r)), 0
        else:
            # `F_out` only moves on acceptance, so `best_loss` only resets once
            # the improvement since it accumulates past `rtol`.
            if float(info["F_out"]) < best_loss * (1.0 - (rtol or 0.0)):
                best_loss, stalled = float(info["F_out"]), 0
            else:
                stalled += 1

        if min_loss and info["F_out"] < min_loss:
            break
        if rtol is not None and stalled >= patience:
            if verbose:
                print(
                    f"loss improved by less than {rtol:g} (relative) in the last "
                    f"{patience} steps; stopping"
                )
            break
    return merge_frozen(theta, frozen), hist
