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


def blocks_of(tree):
    """Parameter blocks: one per conditioner bucket, plus the affine tail."""
    # return list(tree.bijection.bijections[0].inner.conditioners) + [tree.bijection.bijections[1]]
    return [
        *tree.bijection.bijections[0].bijections[0].inner.conditioners,
        (
            tree.bijection.bijections[1],
            #tree.bijection.bijections[0].bijections[1],
        ),
    ]


def rebuild(tree, blocks):
    """Inverse of blocks_of."""
    tree = eqx.tree_at(
        lambda t: t.bijection.bijections[0].bijections[0].inner.conditioners,
        tree,
        tuple(blocks[:-1]),
    )
    tree = eqx.tree_at(lambda t: t.bijection.bijections[1], tree, blocks[-1][0])
    #tree = eqx.tree_at(
    #    lambda t: t.bijection.bijections[0].bijections[1], tree, blocks[-1][1]
    #)
    return tree

    tree = eqx.tree_at(
        lambda t: t.bijection.bijections[0].inner.conditioners, tree, tuple(blocks[:-1])
    )
    return eqx.tree_at(lambda t: t.bijection.bijections[1], tree, blocks[-1])


# ============================================================ sub-blocking


def mlp_unit_labels(one):
    """Depth-1 equinox MLP: label each coordinate by hidden unit, so a sub-block
    is never a single layer (which conditions badly when layer scales differ).
    Returns label arrays in jax.tree.leaves order."""
    l1, l2 = one.layers
    W = l1.weight.shape[0]
    return [
        np.broadcast_to(np.arange(W)[:, None], l1.weight.shape),  # w1 row
        np.arange(W),  # b1
        np.broadcast_to(np.arange(W)[None, :], l2.weight.shape),  # w2 col
        np.arange(l2.bias.shape[0]) % W,
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


def pcg(Av, Minv, b, x0, tol, maxiter):
    """PCG on pytrees.  tol is ABSOLUTE on ||b - Av(x)||.  Returns (x, n_iters)."""

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
        _, r, _, _, rz, k = c
        return (tnorm(r) > tol) & (k < maxiter) & jnp.isfinite(rz)

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

    if capture:
        Gs = jax.lax.map(one, keys, batch_size=batch)
        blocks = [jnp.einsum("tngi,tngj->ngij", G, G) / m for G in Gs]
        return blocks, capture_fractions(Gs, m)

    def chunk_gram(chunk_keys):
        return [
            jnp.einsum("tngi,tngj->ngij", G, G) for G in jax.vmap(one)(chunk_keys)
        ]

    # (n, G, q) per bucket -> (n, G, q, q) accumulators, from shapes alone.
    totals = [
        jnp.zeros(spec.shape + spec.shape[-1:], spec.dtype)
        for spec in jax.eval_shape(one, keys[0])
    ]

    n_chunks, remainder = divmod(m, batch)
    if n_chunks:

        def accumulate(carry, chunk_keys):
            return [c + g for c, g in zip(carry, chunk_gram(chunk_keys))], None

        totals, _ = jax.lax.scan(
            accumulate, totals, keys[: n_chunks * batch].reshape(n_chunks, batch)
        )
    if remainder:
        tail = chunk_gram(keys[n_chunks * batch :])
        totals = [c + g for c, g in zip(totals, tail)]

    return [total / m for total in totals], None


def block_diagonal(blocks, plans, template):
    """`diag(J^T J)` as a pytree shaped like `template`.

    Free: `build_blocks` already estimates each block's Gram matrix, and its
    diagonal is exactly the corresponding run of `diag(J^T J)` -- a plan's
    column index is a permutation, so every live parameter sits in exactly one
    sub-block, in exactly one position.
    """
    out = [
        p["unflatten"](_scatter(jnp.diagonal(H, axis1=-2, axis2=-1), p))
        for p, H in zip(plans, blocks)
    ]
    # `unflatten` recombines each block's static leaves, which `template` (a
    # filtered params pytree) does not carry; drop them again so the result can
    # be tree_map'd against tangents leaf-for-leaf.
    return eqx.filter(rebuild(template, out), eqx.is_inexact_array)


# Fraction of the largest curvature below which Marquardt damping is floored.
# Must not be tiny: `zero_init` sets each conditioner's last layer to zero, so
# the first-layer parameters start with *exactly* zero gradient and curvature,
# and an unfloored `lam * diag(J^T J)` leaves them undamped -- both in CG and,
# far worse, in the block preconditioner, whose inverse then carries entries of
# order `1 / (lam * floor)`.
#
# Measured on the 10-dim funnel (arrow pattern, 10 LM steps, final F / steps
# accepted): 1e-8 -> 1.6e+02, 1/10 (diverges); 1e-4 -> 1.4e-02, 8/10;
# 1e-2 -> 4.8e-02, 9/10; 1e-1 -> 8.9e-02, 10/10. Too low and the flat
# directions blow up; too high and this degrades towards absolute damping,
# which is what it exists to avoid.
MARQUARDT_FLOOR = 1e-4


def marquardt_floor(D):
    """Smallest damping scale allowed, from the largest curvature seen.

    Marquardt damping is `lam * diag(J^T J)`, which vanishes wherever the
    curvature does -- a conditioner with `in_features=0` has a draw-independent
    Jacobian, so its Gram block is exactly zero and an unfloored `lam * diag`
    would leave that block undamped and singular. Flooring at `1e-8` of the
    global maximum only bites on blocks that far below scale, so it rescues the
    degenerate ones without flattening the per-block scaling everywhere else.
    """
    # Empty leaves are skipped: the zero-parent conditioner's first layer has
    # `in_features=0`, so its weight is a genuine zero-size array, and `max`
    # over it has no identity.
    leaves = [l for l in jax.tree.leaves(D) if eqx.is_inexact_array(l) and l.size]
    if not leaves:
        return jnp.asarray(1e-300)
    dmax = jnp.maximum(jnp.max(jnp.stack([jnp.max(leaf) for leaf in leaves])), 1e-300)
    return MARQUARDT_FLOOR * dmax


def _block_inv(H, lam, floor):
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
    if floor is None:
        damped = H + (lam + 1e-12 * jnp.maximum(jnp.mean(jnp.diagonal(H)), 1.0)) * eye
    else:
        diag = lam * jnp.maximum(jnp.diagonal(H), floor) + 1e-4 * floor
        damped = H + jnp.diag(diag)
    factor = jnp.linalg.cholesky(damped)
    inv_factor = solve_triangular(factor, eye, lower=True)
    return inv_factor.T @ inv_factor


def precompute_Minvs(blocks, lam, floor):
    """(n, G, q, q) per bucket.  Computed once per lm_step call and reused
    across every Minv(v) call inside that step's PCG loop.

    `lam` and `floor` are closed over rather than passed through `in_axes`, so
    that `floor=None` (absolute damping) stays a plain trace-time branch inside
    `_block_inv` instead of something vmap has to map over.
    """
    vinv = jax.vmap(jax.vmap(lambda H: _block_inv(H, lam, floor)))
    return [vinv(H) for H in blocks]


def apply_Minvs(Minvs, plans, v):
    out = []
    for b, p, Mi in zip(blocks_of(v), plans, Minvs):
        S = _gather(p["flatten"](b), p)  # (n, G, q)
        S = jnp.einsum("ngij,ngj->ngi", Mi, S)
        out.append(p["unflatten"](_scatter(S, p)))
    return rebuild(v, out)


@eqx.filter_jit
def step(
    res_fn,
    args,
    plans,
    theta,
    r,
    lam,
    key,
    *,
    rebuild_blocks: jax.Array,
    p_prev: jax.Array,
    m=256,
    batch=32,
    precondition=True,
    cg_tol=1e-2,
    cg_eta_max=0.5,
    cg_max=300,
    accept_rho=0.1,
    good_rho=0.75,
    lam_down=3.0,
    lam_up=4.0,
    lam_min=None,
    lam_max=1e10,
    damping="marquardt",
    capture_diagnostic=False,
    blocks=None,
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
    """
    res_fn_args = lambda p: res_fn(p, args)
    _, vjpf = jax.vjp(res_fn_args, theta)
    vjp = lambda w: vjpf(w)[0]

    if damping not in ("marquardt", "absolute"):
        raise ValueError(
            f"Unknown damping {damping!r}, expected 'marquardt' or 'absolute'."
        )
    if lam_min is None:
        # Under Marquardt damping `lam` is dimensionless -- a fraction of each
        # block's own curvature -- so the 1e-10 that made sense for absolute
        # damping means "undamped to well below double precision", and CG ends
        # up solving a near-singular system to a tolerance it cannot reach.
        lam_min = 1e-6 if damping == "marquardt" else 1e-10

    if precondition:
        if blocks is None or capture_diagnostic:
            # Nothing to reuse on the first step, and the diagnostic needs the
            # individual probes, so both always build.
            Hb, capture = build_blocks(
                vjpf, r.shape, key, plans, m, batch, capture=capture_diagnostic
            )
        else:
            capture = None
            Hb = jax.lax.cond(
                rebuild_blocks,
                lambda: build_blocks(vjpf, r.shape, key, plans, m, batch)[0],
                lambda: blocks,
            )
        if damping == "marquardt":
            D = block_diagonal(Hb, plans, theta)
            floor = marquardt_floor(D)
            D = jax.tree.map(lambda d: jnp.maximum(d, floor), D)
        else:
            D = floor = None
        Minvs = precompute_Minvs(Hb, lam, floor)
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

    # Inexact-Newton forcing term (Dembo/Eisenstat/Steihaug; the
    # `min(eta_max, sqrt(||g||))` rule from the trust-region Newton line).
    # `eta -> 0` as the gradient does, which keeps the superlinear rate, while
    # staying loose early where an accurate solve buys nothing. It reads only
    # the current gradient, so it carries no state and is unchanged across a
    # rejected step -- it cannot interact with the `lam` update.
    #
    # `cg_tol` is the floor: setting `cg_eta_max = cg_tol` recovers the old
    # fixed-tolerance behaviour exactly.
    residual_norm = tnorm(rhs)
    if False:
        # Fixed relative tolerance. Near convergence this shrinks in lockstep
        # with the gradient while the system gets no easier, so CG burns
        # `cg_max` on steps whose truncated solution already gives rho ~ 1.
        eta = cg_tol
    else:
        eta = jnp.clip(jnp.sqrt(residual_norm), cg_tol, cg_eta_max)

    p, ncg = pcg(Av, Minv, rhs, x0, eta * residual_norm, cg_max)

    theta_new = jax.tree.map(jnp.add, theta, p)
    r_new = res_fn_args(theta_new)
    Jp = jvp(p)

    F, F_new = tdot(r, r), tdot(r_new, r_new)
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
    lam_out = jnp.clip(
        jnp.where(accept, jnp.where(rho > good_rho, lam / lam_down, lam), lam * lam_up),
        lam_min,
        lam_max,
    )
    # a rejected step's p solves a system with a different lambda: discard it
    p_out = jax.tree.map(lambda a: jnp.where(accept, a, jnp.zeros_like(a)), p)

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
        "n_cg": ncg,
        "cg_eta": eta,
        "cg_converged": ncg < cg_max,
        "grad_norm": tnorm(g),
        "step_norm": tnorm(p),
        "finite": ok,
        "rebuilt_blocks": rebuild_blocks,
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
        bs = blocks_of(params)
        label_fns = [mlp_unit_labels] * (len(bs) - 1) + [
            None,
            #None,
        ]  # affine tail: no units

        _plans_cache[key] = [make_plan(b, m, lf, q_min) for b, lf in zip(bs, label_fns, strict=True)]
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
    batch=32,
    damping="marquardt",
    lam_min=None,
    capture_diagnostic=False,
    rebuild_every=3,
):
    """Levenberg-Marquardt fit.

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
    """
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

    theta, r, lam = params, res_fn(params, args), jnp.asarray(lam0)
    # Seed with real zeros (not None) so p_prev's pytree structure is the
    # same on every iteration -- else step 0 (p_prev=None) and step 1
    # (p_prev=<pytree>) are different input structures and each triggers its
    # own trace of lm_step (and the pcg while_loop inside it).
    p_prev = jax.tree.map(jnp.zeros_like, params)
    key, hist = jr.key(seed), []
    # The Gram estimate is a function of `theta` alone, so it stays exactly
    # valid across a rejected step -- `theta` did not move. Between accepted
    # steps it only drifts, so `rebuild_every` trades a slightly stale
    # preconditioner (which costs CG iterations, never correctness) against `m`
    # reverse passes. Counting only accepted steps makes the reuse on rejection
    # exact rather than an approximation.
    blocks = None
    accepted_since_build = rebuild_every  # build on the first step
    for i in range(n_steps):
        key, sk = jr.split(key)
        rebuild_blocks = accepted_since_build >= rebuild_every
        theta, r, lam, p_prev, blocks, info = step(
            res_fn,
            args,
            plans,
            theta,
            r,
            lam,
            sk,
            m=m,
            precondition=precondition,
            p_prev=p_prev,
            cg_max=cg_max,
            cg_eta_max=cg_eta_max,
            batch=batch,
            damping=damping,
            lam_min=lam_min,
            capture_diagnostic=capture_diagnostic,
            blocks=blocks,
            rebuild_blocks=jnp.asarray(rebuild_blocks),
        )
        if rebuild_blocks:
            accepted_since_build = 0
        accepted_since_build += int(info["accept"])
        hist.append(info)
        if verbose:
            print(
                f"{i:3d}  F={float(info['F_new']):.4e}  "
                f"log(F)={float(np.log(info['F_new'])):+.2f} "
                f"rho={float(info['rho']):+.2f}  "
                f"lam={float(info['lam_out']):.1e}  "
                f"cg={int(info['n_cg']):3d}{'' if info['cg_converged'] else '*'} "
                f"eta={float(info['cg_eta']):.2f}"
                f"{' ' if info['rebuilt_blocks'] else '~'}  "
                f"|g|={float(info['grad_norm']):.2e}"
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
            theta = jax.tree.map(lambda x: 1e-2 * jr.normal(sk, x.shape), theta)

        if min_loss and info["F_out"] < min_loss:
            break
    return theta, hist
