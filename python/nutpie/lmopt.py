import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular

# ============================================================ pytree helpers


def tdot(a, b):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def tnorm(a):
    return jnp.sqrt(tdot(a, a))


def codec(block):
    """``(flatten, unflatten, Pb)`` for one bucket, ``flatten: block -> (n, Pb)``."""
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
    """One block per conditioner bucket, plus the affine tail if not frozen."""
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
    """``(trainable, frozen)``: the affine tail split off as `frozen`, unless
    `fit_affine` (then `frozen` is ``None``).

    The tail is nearly redundant with the conditioners' own shift and scale,
    and fitting it adds coupling the block preconditioner cannot see.
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
    """Label each parameter of a depth-1 MLP by hidden unit, in
    `jax.tree.leaves` order, so sub-blocks span layers. Skip weights of a
    `LocationSkipMlp` get one extra label."""
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
    assert [tuple(l.shape) for l in leaves] == [tuple(np.shape(g)) for g in labels], (
        "label shapes must match jax.tree.leaves order of the block"
    )
    lab = np.concatenate([np.asarray(g).ravel() for g in labels])
    return np.argsort(lab, kind="stable")


def block_limit(m, q_min=16, max_block_size=None):
    """Largest sub-block size: `max_block_size` if given (exact blocks,
    where only the cost matters), else ``max(q_min, 2 m)`` (probe-based
    blocks, which `m` probes can only estimate up to a certain size)."""
    return max_block_size if max_block_size is not None else max(q_min, 2 * m)


def describe_plans(plans, limit):
    """Table of the blocks of each bucket, for the verbose output.

    `parents` is the input width of the bucket's conditioners, the largest
    number of parents in the bucket.
    """
    lines = [
        f"LM preconditioner blocks (sub-blocks of at most {limit} parameters):",
        "  parents  conditioners  weights/conditioner  blocks/conditioner",
    ]
    for p in plans:
        parents = "affine" if p["parents"] is None else p["parents"]
        blocks = f"{p['G']} x {p['q']}"
        lines.append(f"  {parents:>7}  {p['n']:>12}  {p['Pb']:>19}  {blocks:>18}")
    return "\n".join(lines)


def describe_fallbacks(info, plans):
    """Sub-blocks replaced by their diagonal in this step, for the verbose
    output, e.g. ``"  non-finite blocks: 3 (8 parents: 3)"``."""
    text = ""
    for key, label in [
        ("nonfinite_blocks", "non-finite blocks"),
        ("failed_inverses", "diagonal inverses"),
    ]:
        counts = np.asarray(info.get(key, []))
        if not counts.sum():
            continue
        buckets = ", ".join(
            f"{'affine' if p['parents'] is None else p['parents']} parents: {c}"
            for c, p in zip(counts, plans)
            if c
        )
        text += f"  {label}: {counts.sum()} ({buckets})"
    return text


def mlp_input_width(one):
    """Input width of one conditioner, as given to `make_net`."""
    from nutpie.triangular import LocationSkipMlp

    mlp = one.mlp if isinstance(one, LocationSkipMlp) else one
    return int(mlp.layers[0].weight.shape[-1])


def make_plan(block, m, label_fn=None, q_min=16, max_block_size=None):
    """Sub-block plan for one bucket: ``G`` sub-blocks of size
    ``q = min(Pb, block_limit(m, q_min, max_block_size))``."""
    flatten, unflatten, Pb = codec(block)
    arrays, _ = eqx.partition(block, eqx.is_inexact_array)
    n = jax.tree.leaves(arrays)[0].shape[0]
    # Conditioners have labels, the affine tail doesn't.
    parents = (
        mlp_input_width(jax.tree.map(lambda leaf: leaf[0], block))
        if label_fn is not None
        else None
    )
    perm = (
        _perm_from_labels(block, label_fn)
        if (label_fn is not None and Pb > 0)
        else np.arange(Pb)
    )
    # q = int(min(Pb, max(q_min, m // 2))) if Pb else 1
    q = int(min(Pb, block_limit(m, q_min, max_block_size))) if Pb else 1
    G = -(-Pb // q) if Pb else 1
    idx = np.full(G * q, Pb, dtype=int)
    idx[:Pb] = perm  # column Pb = dummy sink
    return {
        "flatten": flatten,
        "unflatten": unflatten,
        "n": n,
        "parents": parents,
        "Pb": Pb,
        "q": q,
        "G": G,
        "idx": jnp.asarray(idx.reshape(G, q)),
        "mask": jnp.asarray((idx < Pb).reshape(G, q)).astype(float),
    }


def _gather(A, p):
    """(..., n, Pb) -> (..., n, G, q), padded entries zeroed."""
    A = jnp.concatenate([A, jnp.zeros(A.shape[:-1] + (1,), A.dtype)], -1)
    return A[..., p["idx"]] * p["mask"]


def _scatter(S, p):
    """(n, G, q) -> (n, Pb)."""
    n = S.shape[0]
    flat = jnp.zeros((n, p["Pb"] + 1), S.dtype).at[:, p["idx"]].add(S * p["mask"])
    return flat[:, : p["Pb"]]


# ============================================================ preconditioned CG


def pcg(Av, Minv, b, x0, rtol, maxiter):
    """PCG on pytrees, returning ``(x, n_iters)``.

    Stops once ``||r||_{M^-1} < rtol ||b||_{M^-1}``: free, since ``r . z`` is
    formed anyway, and scaled by each block's own curvature.
    """
    tol_sq = rtol**2 * tdot(b, Minv(b))

    Ax0 = Av(x0)
    bx, xAx = tdot(b, x0), tdot(x0, Ax0)
    # Best multiple of the warm start, so the model value starts <= 0;
    # alpha = 0 if x0 is uphill or zero.
    alpha = jnp.where((bx > 0) & (xAx > 0), bx / jnp.where(xAx > 0, xAx, 1.0), 0.0)
    x0 = jax.tree.map(lambda x: alpha * x, x0)
    r0 = jax.tree.map(lambda bb, ax: bb - alpha * ax, b, Ax0)
    z0 = Minv(r0)

    # r0 = jax.tree.map(lambda p, q: p - q, b, Av(x0))
    # z0 = Minv(r0)

    def cond(c):
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

    x, _, _, _, _, k = jax.lax.while_loop(cond, body, (x0, r0, z0, z0, tdot(r0, z0), 0))
    return x, k


def capture_fractions(Gs, m):
    """Fraction of ``||J^T J||_F^2`` inside the sub-blocks (``sub_block``) and
    inside whole conditioners (``conditioner``); the rest is coupling between
    conditioners.

    Uses ``tr(Hhat_1 Hhat_2)`` over two independent halves of the probes,
    which is unbiased. Estimates, so they can exceed 1.
    """
    half = m // 2
    sub_mass = jnp.zeros(())
    conditioner_mass = jnp.zeros(())
    total_cross = jnp.zeros((half, half))
    for G in Gs:
        first, second = G[:half], G[half : 2 * half]
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
    """Hutchinson estimate of the ``(n, G, q, q)`` blocks per bucket, from `m`
    Rademacher VJPs through `vjpf`, `batch` at a time. Unbiased and PSD.

    `capture` keeps all ``m`` probes in memory for `capture_fractions`.
    """

    def one(k):
        w = jr.rademacher(k, shape).astype("float64")
        g = vjpf(w)[0]
        return [_gather(p["flatten"](b), p) for b, p in zip(blocks_of(g), plans)]

    keys = jr.split(key, m)
    totals, capture_info = _accumulate_grams(one, keys, batch, capture)
    return [total / m for total in totals], capture_info


def _accumulate_grams(one, items, batch, capture):
    """``(totals, capture)``: ``sum_t G_t^T G_t`` per bucket with
    ``G_t = one(items[t])``, `batch` items at a time."""
    n_items = jax.tree.leaves(items)[0].shape[0]

    if capture:
        Gs = jax.lax.map(one, items, batch_size=batch)
        totals = [jnp.einsum("tngi,tngj->ngij", G, G) for G in Gs]
        return totals, capture_fractions(Gs, n_items)

    def chunk_gram(chunk):
        # Leading axes before `(n, G, q)` are summed over.
        return [
            jnp.einsum(
                "tngi,tngj->ngij",
                G.reshape((-1,) + G.shape[-3:]),
                G.reshape((-1,) + G.shape[-3:]),
            )
            for G in jax.vmap(one)(chunk)
        ]

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
    """Exact Gauss-Newton blocks per bucket, ``sum_draws V^T V / n_draws``
    with `factor(draw_data)` giving ``V`` (see
    `SparseTriangularMap.gauss_newton_factors`)."""
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
    """Hutchinson estimate of the blocks from per-group probes.

    Draws are split into `n_groups` contiguous groups, each with its own
    probe, so only draw pairs within a group add noise. One round gives
    `n_groups` samples for the cost of one pass over all draws.

    `group_res_fn(theta, group_data)` must normalize residuals by
    ``1 / sqrt(group size)``. `batch` is groups per chunk. `capture` needs an
    even `rounds`.
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

    # Round-major, so each `capture` half is whole rounds.
    groups = jnp.asarray(np.tile(np.arange(n_groups), rounds))
    keys = jr.split(key, n_groups * rounds)
    totals, capture_info = _accumulate_grams(one, (groups, keys), batch, capture)
    return [total * (size / n) / rounds for total in totals], capture_info


def block_diagonal(blocks, plans, template, floors):
    """``diag(J^T J)`` read off the blocks, floored per conditioner, as a
    pytree shaped like `template`."""
    out = [
        p["unflatten"](
            _scatter(
                jnp.maximum(jnp.diagonal(H, axis1=-2, axis2=-1), floor[:, None, None]),
                p,
            )
        )
        for p, H, floor in zip(plans, blocks, floors)
    ]
    # Drop the static leaves `unflatten` adds, to match tangents leaf-for-leaf.
    return eqx.filter(rebuild(template, out), eqx.is_inexact_array)


# Marquardt damping floor, as a fraction of each conditioner's largest
# curvature. Zero-initialized last layers leave first-layer curvature near
# zero, which would otherwise go undamped; too high approaches absolute
# damping.
MARQUARDT_FLOOR = 1e-4
# Fraction of the global largest curvature, for conditioners flat as a whole.
MARQUARDT_GLOBAL_FLOOR = 1e-10


def marquardt_floors(blocks):
    """Per-conditioner floor for ``diag(J^T J)`` in Marquardt damping,
    ``(n,)`` per bucket."""
    local = [jnp.max(jnp.diagonal(H, axis1=-2, axis2=-1), axis=(1, 2)) for H in blocks]
    global_max = jnp.maximum(jnp.max(jnp.concatenate(local)), 1e-300)
    return [
        jnp.maximum(MARQUARDT_FLOOR * dmax, MARQUARDT_GLOBAL_FLOOR * global_max)
        for dmax in local
    ]


def _sanitize_blocks(blocks):
    """Replace non-finite ``(q, q)`` sub-blocks by their finite diagonal.

    A single non-finite block would otherwise spread through the global
    Marquardt floor into every conditioner's damping, and through `D` into
    the damped system itself, so CG could not make any step. Returns the
    blocks and the number of replaced sub-blocks per bucket.
    """
    out, counts = [], []
    for H in blocks:
        bad = ~jnp.all(jnp.isfinite(H), axis=(-2, -1))
        diag = jnp.diagonal(H, axis1=-2, axis2=-1)
        diag = jnp.where(jnp.isfinite(diag), diag, 0.0)
        fallback = diag[..., :, None] * jnp.eye(H.shape[-1], dtype=H.dtype)
        out.append(jnp.where(bad[..., None, None], fallback, H))
        counts.append(jnp.sum(bad))
    return out, jnp.stack(counts)


def _block_inv(H, lam, floor, shrinkage):
    """Inverse of the damped block ``H + lam D``, and whether it failed.

    ``D = I`` if `floor` is ``None`` (Levenberg), else ``diag(H)`` floored at
    `floor` (Marquardt); it must match `step`'s `Av`. `H` is first shrunk
    towards its diagonal by `shrinkage`, so directions the probes missed fall
    back to Jacobi. A small ridge keeps the Cholesky stable. If the inverse
    is still not finite, the inverse of the damped diagonal is used instead.
    """
    if False:
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
    inv = inv_factor.T @ inv_factor
    failed = ~jnp.all(jnp.isfinite(inv))
    return jnp.where(failed, jnp.diag(1.0 / jnp.diagonal(damped)), inv), failed


def precompute_Minvs(blocks, plans, n_samples, lam, floors):
    """Inverse damped blocks ``(n, G, q, q)`` per bucket, once per step, and
    the number of sub-blocks per bucket that fell back to the diagonal.

    `floors` is ``None`` for absolute damping. Shrinkage is
    ``q / (q + n_samples)``.
    """
    out, failed = [], []
    for i, (H, p) in enumerate(zip(blocks, plans)):
        shrinkage = p["q"] / (p["q"] + n_samples)
        if floors is None:
            inv = lambda H: _block_inv(H, lam, None, shrinkage)
            Minv, bad = jax.vmap(jax.vmap(inv))(H)
        else:
            inv = lambda H, floor: _block_inv(H, lam, floor, shrinkage)
            Minv, bad = jax.vmap(jax.vmap(inv, in_axes=(0, None)))(H, floors[i])
        out.append(Minv)
        failed.append(jnp.sum(bad))
    return out, jnp.stack(failed)


def apply_Minvs(Minvs, plans, v):
    out = []
    for b, p, Mi in zip(blocks_of(v), plans, Minvs):
        S = _gather(p["flatten"](b), p)  # (n, G, q)
        S = jnp.einsum("ngij,ngj->ngi", Mi, S)
        out.append(p["unflatten"](_scatter(S, p)))
    return rebuild(v, out)


def blocks_zeros(theta, plans):
    """Zeros shaped like `build_blocks`' output, so the first `step` call has
    the same argument structure as later ones and does not retrace."""
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
    lam_lo_in: jax.Array,
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
    lam_lo_decay=3.0**0.5,
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
    """One LM step. `res_fn` and `plans` are static; get `plans` from
    `get_plans` so its identity is stable across calls.

    `blocks` is the previous step's Gram estimate, reused unless the traced
    `rebuild_blocks` is set. `eta_in`, `nu_in` and `lam_lo_in` are carried
    state, returned as ``info["cg_eta_next"]``, ``info["lam_nu_next"]`` and
    ``info["lam_lo_next"]``. `theta` excludes `frozen` (see `split_frozen`).

    Blocks come from `factor_fn` if given, else `build_blocks` (`n_groups`
    ``None``), else `build_blocks_grouped`.
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
        # Marquardt `lam` is relative to each block's curvature.
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
                group_res_fn,
                data,
                theta,
                key,
                plans,
                n_groups,
                rounds,
                batch,
                capture=capture,
            )
        if blocks is None or capture_diagnostic:
            Hb, capture = estimate(capture_diagnostic)
        else:
            capture = None
            Hb = jax.lax.cond(
                rebuild_blocks, lambda: estimate(False)[0], lambda: blocks
            )
        Hb, nonfinite_blocks = _sanitize_blocks(Hb)
        if damping == "marquardt":
            floors = marquardt_floors(Hb)
            D = block_diagonal(Hb, plans, theta, floors)
        else:
            D = floors = None
        Minvs, failed_inverses = precompute_Minvs(Hb, plans, n_samples, lam, floors)
        Minv = lambda v: apply_Minvs(Minvs, plans, v)
    else:
        # No blocks, so no diagonal for Marquardt damping.
        Hb = D = capture = None
        nonfinite_blocks = failed_inverses = None
        Minv = lambda v: v

    jvp = lambda v: jax.jvp(res_fn_args, (theta,), (v,))[1]
    if D is None:
        Av = lambda v: jax.tree.map(lambda a, b: a + lam * b, vjp(jvp(v)), v)
    else:
        Av = lambda v: jax.tree.map(lambda a, d, b: a + lam * d * b, vjp(jvp(v)), D, v)

    g = vjp(r)
    rhs = jax.tree.map(jnp.negative, g)
    x0 = p_prev

    # Equal `cg_tol` and `cg_eta_max` give a fixed tolerance.
    eta = jnp.clip(eta_in, cg_tol, cg_eta_max)

    p, ncg = pcg(Av, Minv, rhs, x0, eta, cg_max)

    theta_new = jax.tree.map(jnp.add, theta, p)
    r_new = res_fn_args(theta_new)
    Jp = jvp(p)

    F, F_new = tdot(r, r), tdot(r_new, r_new)
    step_length = jnp.ones((), F.dtype)
    # `lam` is driven by the full GN step, not the line-searched one.
    pred_full = -tdot(p, g) - 0.5 * tdot(Jp, Jp)
    rho_full = (
        0.5 * (F - F_new) / jnp.where(jnp.abs(pred_full) < 1e-30, 1e-30, pred_full)
    )
    full_step_good = (
        (rho_full > accept_rho)
        & (pred_full > 0)
        & jnp.isfinite(F_new)
        & jnp.isfinite(rho_full)
    )
    full_step_norm = tnorm(p)

    if line_search:
        # On large-residual problems GN underestimates curvature and
        # overshoots. Fit a parabola through f(0), f'(0) and f(1) along `p`.
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
    # `pred > 0` guards against both signs flipping into a positive `rho`.
    accept = (rho > accept_rho) & (pred > 0) & ok

    pick = lambda a, b: jax.tree.map(lambda x, y: jnp.where(accept, x, y), a, b)
    theta_out, r_out = pick(theta_new, theta), pick(r_new, r)

    # Nielsen's update: the decrease is graded by `rho` (raising `lam` below
    # `rho = 0.5`), and `nu` doubles on each consecutive failure.
    lam_decrease = jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho_full - 1.0) ** 3)
    # Decrease no further than the log-midpoint to `lam_lo`, the last `lam`
    # that overshot, to avoid a period-2 cycle with the line search.
    # `lam_lo` decays on clean steps.
    lam_decreased = jnp.maximum(
        lam * lam_decrease, jnp.minimum(lam, jnp.sqrt(lam * lam_lo_in))
    )
    lam_next = jnp.where(full_step_good, lam_decreased, lam * nu_in)
    nu_out = jnp.where(full_step_good, nu0, 2.0 * nu_in)
    lam_lo_out = jnp.where(full_step_good, lam_lo_in / lam_lo_decay, lam_lo_in)
    # After a shortened step of fraction `a`, `lam / a` makes the next GN
    # step about the length the line search found.
    shortened = accept & (step_length < 1.0)
    lam_next = jnp.where(shortened, lam / step_length, lam_next)
    nu_out = jnp.where(shortened, nu0, nu_out)
    lam_lo_out = jnp.where(shortened, lam, lam_lo_out)
    lam_out = jnp.clip(lam_next, lam_min, lam_max)
    # A rejected `p` was solved for a different `lam`: discard it.
    p_out = jax.tree.map(lambda a: jnp.where(accept, a, jnp.zeros_like(a)), p)

    # CG forcing term for the next step, from how well the linear model
    # predicted this one:
    #   "residual": | ||r_new|| - ||r + J p|| | / ||r||  (Eisenstat-Walker 1)
    #   "rho":      |1 - rho_full|, which keeps adapting when F plateaus
    #               well above zero.
    if forcing == "rho":
        eta_next = jnp.abs(1.0 - rho_full)
        # Non-finite `rho`: ask for the loosest solve.
        eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, cg_eta_max)
    else:
        linear_r_norm = tnorm(jax.tree.map(jnp.add, r, Jp))
        eta_next = jnp.abs(jnp.sqrt(F_new) - linear_r_norm) / jnp.sqrt(
            jnp.where(F > 0, F, 1.0)
        )
    # Eisenstat-Walker safeguard against decreasing `eta` too fast.
    safeguard = cg_gamma * eta**cg_alpha
    eta_next = jnp.where(safeguard > 0.1, jnp.maximum(eta_next, safeguard), eta_next)
    # Hold `eta` on rejection.
    eta_next = jnp.where(accept & ok, eta_next, eta)

    info = {
        "F": F,
        "F_new": F_new,
        # Loss of the carried-forward state; `F_new` is the proposal.
        "F_out": jnp.where(accept, F_new, F),
        "accept": accept,
        "rho": rho,
        # Of the full GN step, before the line search; drives `lam`.
        "rho_full": rho_full,
        "actual": actual,
        "pred": pred,
        "lam_in": lam,
        "lam_out": lam_out,
        "lam_nu": nu_in,
        "lam_nu_next": nu_out,
        "lam_lo_next": lam_lo_out,
        "n_cg": ncg,
        "cg_eta": eta,
        "cg_eta_next": eta_next,
        "cg_converged": ncg < cg_max,
        "grad_norm": tnorm(g),
        "step_norm": tnorm(p),
        # The line search shortens this to `step_length * full_step_norm`.
        "full_step_norm": full_step_norm,
        "finite": ok,
        "rebuilt_blocks": rebuild_blocks,
        "step_length": step_length,
    }
    if capture is not None:
        info["capture"] = capture
    if nonfinite_blocks is not None:
        info["nonfinite_blocks"] = nonfinite_blocks
        info["failed_inverses"] = failed_inverses
    # Block eigenvalue diagnostics, disabled.
    if False and plans is not None:
        info["block_min_eig"] = jnp.stack([jnp.min(jnp.linalg.eigvalsh(H)) for H in Hb])
        info["block_max_eig"] = jnp.stack([jnp.max(jnp.linalg.eigvalsh(H)) for H in Hb])
    return theta_out, r_out, lam_out, p_out, Hb, info


# ============================================================ setup and driver

_plans_cache = {}


def _plans_signature(params, m, q_min, max_block_size):
    """`_plans_cache` key: treedef, leaf shapes and dtypes, plan settings."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = tuple((tuple(np.shape(l)), np.result_type(l).str) for l in leaves)
    return (treedef, shapes, m, q_min, max_block_size)


def get_plans(params, m=128, q_min=16, max_block_size=None):
    """Block plans for this parameter structure, cached so that `step`, which
    takes them as static arguments, does not recompile."""
    key = _plans_signature(params, m, q_min, max_block_size)
    if key not in _plans_cache:
        n_conditioners = len(_conditioners(params))
        _plans_cache[key] = [
            # The affine tail has no hidden units to group by.
            make_plan(
                b,
                m,
                mlp_unit_labels if i < n_conditioners else None,
                q_min,
                max_block_size,
            )
            for i, b in enumerate(blocks_of(params))
        ]
    return _plans_cache[key]


def setup(params, m=128, q_min=16, max_block_size=None):
    """Block plans for `step`, checking that `blocks_of` round-trips."""
    plans = get_plans(params, m, q_min, max_block_size)

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
    max_exact_block_size=256,
    print_blocks=False,
    should_stop=None,
):
    """Levenberg-Marquardt fit of ``res_fn(params, (*args, *data))``.

    `data` holds per-draw arrays (leading axis: draws). Stops after `n_steps`,
    below `min_loss`, or when `patience` steps (rejections included) lowered
    the loss by less than a fraction `rtol`.

    Args:
        m: Probes per block estimate; also sets the sub-block size.
        batch: Probes (or groups) evaluated at once; the main memory knob.
        damping: ``"marquardt"`` (``lam * diag(J^T J)``, `lam` relative) or
            ``"absolute"`` (``lam * I``). `lam0` is not comparable across them.
        n_groups, rounds: Use `build_blocks_grouped` instead of `build_blocks`.
        factor_fn: ``factor_fn(params, args, draw_data)`` gives exact block
            factors (see `build_blocks_exact`). Needs ``fit_affine=False``.
        max_exact_block_size: Largest sub-block with exact blocks. Larger
            conditioners are split, only to bound the cost; with probe-based
            blocks the size follows from `m` instead.
        fit_affine: Also fit the affine tail, see `split_frozen`.
        line_search: Shorten steps to a fitted parabola's minimum (``a=`` in
            the log).
        forcing: CG tolerance rule, ``"residual"`` or ``"rho"``; see `step`.
        rebuild_every: Accepted steps between block estimates.
        verbose: Print one line per step.
        print_blocks: Print the preconditioner blocks.
        should_stop: Called after each step; the fit stops early if it
            returns true, e.g. when sampling is aborted.
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
        max_block_size = max_exact_block_size if factor_fn is not None else None
        plans = setup(params, m, max_block_size=max_block_size)

        if print_blocks:
            print(describe_plans(plans, block_limit(m, max_block_size=max_block_size)))
    else:
        plans = None

    key = jax.random.key(0)

    theta, r = params, res_fn(merge_frozen(params, frozen), (*args, *data))
    # Carried state is seeded with `step`'s output structure and strong dtypes
    # (zeros, not None; `r.dtype`, not Python floats) to avoid a retrace.
    # `lam_lo = 0` leaves the first decrease unbounded.
    p_prev = jax.tree.map(jnp.zeros_like, params)
    best_loss, stalled = float(tdot(r, r)), 0
    lam = jnp.asarray(lam0, dtype=r.dtype)
    eta_prev = jnp.asarray(cg_eta_max, dtype=r.dtype)
    nu_prev = jnp.asarray(nu0, dtype=r.dtype)
    lam_lo_prev = jnp.zeros((), dtype=r.dtype)
    key, hist = jr.key(seed), []
    # Blocks depend only on `theta`, so only accepted steps count towards
    # `rebuild_every`. The zeros are never read.
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
            lam_lo_in=lam_lo_prev,
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
        lam_lo_prev = info["lam_lo_next"]
        hist.append(info)
        if should_stop is not None and should_stop():
            break
        if verbose:
            print(
                f"{i:3d}  log F={float(np.log(info['F_new'])):+.2f}  "
                f"rho={float(info['rho']):+.2f}  "
                f"rho_full={float(info['rho_full']):+.2f}  "
                f"lam={float(info['lam_out']):.1e}  "
                f"cg={int(info['n_cg']):3d}{' ' if info['cg_converged'] else '*'} "
                f"eta={float(info['cg_eta']):.2f}"
                f"{' ' if info['rebuilt_blocks'] else '~'}  "
                f"|g|={float(info['grad_norm']):.2e}  "
                f"|p|={float(info['full_step_norm']):.2e}"
                + (f"  a={float(info['step_length']):.2f}" if line_search else "")
                + (
                    "  capture sub/cond="
                    f"{float(info['capture']['sub_block']):.2f}/"
                    f"{float(info['capture']['conditioner']):.2f}"
                    if "capture" in info
                    else ""
                )
                + describe_fallbacks(info, plans)
                + f"{'' if info['accept'] else '   REJECT'}"
            )

        if not np.isfinite(info["grad_norm"]):
            if verbose:
                print("gradient norm is NaN or Inf; adding noise to parameters")
            # Jitter inexact leaves, one key per leaf.
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
            # Reset all state that refers to the old `theta`.
            r = res_fn(merge_frozen(theta, frozen), (*args, *data))
            p_prev = jax.tree.map(jnp.zeros_like, theta)
            accepted_since_build = rebuild_every
            eta_prev = jnp.asarray(cg_eta_max, dtype=r.dtype)
            nu_prev = jnp.asarray(nu0, dtype=r.dtype)
            lam_lo_prev = jnp.zeros((), dtype=r.dtype)
            best_loss, stalled = float(tdot(r, r)), 0
        else:
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
