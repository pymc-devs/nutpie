import jax, jax.numpy as jnp, numpy as np, equinox as eqx
from jax import random as jr
from jax.flatten_util import ravel_pytree


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
    return list(tree.bijection.bijections[0].inner.conditioners) + [tree.bijection.bijections[1]]


def rebuild(tree, blocks):
    """Inverse of blocks_of."""
    tree = eqx.tree_at(lambda t: t.bijection.bijections[0].inner.conditioners,
                       tree, tuple(blocks[:-1]))
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
    q = int(min(Pb, max(q_min, m // 2))) if Pb else 1
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
    r0 = jax.tree.map(lambda p, q: p - q, b, Av(x0))
    z0 = Minv(r0)

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


# ============================================================ build and apply
#
# These used to be factories that returned a fresh closure on every call
# (make_block_builder/make_apply -> build/Minv).  Those closures were then
# passed into `lm_step` as arguments, and since `setup` (and therefore these
# factories) ran again on every `fit` call, `step = eqx.filter_jit(lm_step)`
# saw a brand-new, unequal static argument each time and recompiled even when
# nothing about the problem had changed.  They're now plain functions that
# `lm_step` calls directly; the only closures that remain (`Minv` below) are
# created *inside* the traced function, so they never cross the jit boundary
# and can't trigger a recompile.


def build_blocks(vjpf, shape, key, plans, m, batch=32):
    """[ (n_j, G_j, q_j, q_j) ] per bucket, estimated from m VJPs.

    E[(G^T w)(G^T w)^T] = G^T G for isotropic w, so a single VJP updates every
    block at once and the estimate is unbiased and PSD by construction.
    `vjpf` is the caller's already-linearized res_fn (see lm_step) -- forming
    it here too would make XLA build and optimize a second, identical copy of
    the residual function's forward+backward graph, which measurably bloats
    compile time.
    """

    def one(k):
        w = jr.rademacher(k, shape).astype("float32")
        g = vjpf(w)[0]
        return [_gather(p["flatten"](b), p) for b, p in zip(blocks_of(g), plans)]

    Gs = jax.lax.map(one, jr.split(key, m), batch_size=batch)
    return [jnp.einsum("tngi,tngj->ngij", G, G) / m for G in Gs]


def _block_inv(H, lam):
    """eigh with clamping rather than Cholesky: blocks are small, some are
    structurally singular (a conditioner with in_features=0 has a
    draw-independent Jacobian), and eigh avoids the static `lower` flag that
    breaks under vmap."""
    w, V = jnp.linalg.eigh(H)
    return jnp.einsum("ij,j,kj->ik", V, 1.0 / (jnp.maximum(w, 0.0) + lam), V)


def precompute_Minvs(blocks, lam):
    """(n, G, q, q) per bucket.  Computed once per lm_step call and reused
    across every Minv(v) call inside that step's PCG loop."""
    vinv = jax.vmap(jax.vmap(_block_inv, in_axes=(0, None)), in_axes=(0, None))
    return [vinv(H, lam) for H in blocks]


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
    m=128,
    batch=32,
    precondition=True,
    cg_tol=1e-2,
    cg_max=300,
    p_prev=None,
    accept_rho=0.1,
    good_rho=0.75,
    lam_down=3.0,
    lam_up=4.0,
    lam_min=1e-10,
    lam_max=1e10,
):
    """One LM step.  res_fn and plans are static under eqx.filter_jit; plans
    is built once via get_plans and cached by parameter structure, so its
    identity (and that of the flatten/unflatten closures it holds) stays
    stable across steps and across repeated `fit` calls with the same
    architecture."""
    res_fn_args = lambda p: res_fn(p, args)
    _, vjpf = jax.vjp(res_fn_args, theta)
    vjp = lambda w: vjpf(w)[0]

    if precondition:
        Hb = build_blocks(vjpf, r.shape, key, plans, m, batch)
        Minvs = precompute_Minvs(Hb, lam)
        Minv = lambda v: apply_Minvs(Minvs, plans, v)
    else:
        Hb = None
        Minv = lambda v: v

    jvp = lambda v: jax.jvp(res_fn_args, (theta,), (v,))[1]
    Av = lambda v: jax.tree.map(lambda a, b: a + lam * b, vjp(jvp(v)), v)

    g = vjp(r)
    rhs = jax.tree.map(jnp.negative, g)
    x0 = jax.tree.map(jnp.zeros_like, theta) if p_prev is None else p_prev

    p, ncg = pcg(Av, Minv, rhs, x0, cg_tol * tnorm(rhs), cg_max)

    theta_new = jax.tree.map(jnp.add, theta, p)
    r_new = res_fn_args(theta_new)
    Jp = jvp(p)

    F, F_new = tdot(r, r), tdot(r_new, r_new)
    actual = 0.5 * (F - F_new)
    pred = -tdot(p, g) - 0.5 * tdot(Jp, Jp)  # undamped GN model
    rho = actual / jnp.where(jnp.abs(pred) < 1e-30, 1e-30, pred)

    ok = jnp.isfinite(F_new) & jnp.isfinite(rho)
    accept = (rho > accept_rho) & ok

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
        "accept": accept,
        "rho": rho,
        "actual": actual,
        "pred": pred,
        "lam_in": lam,
        "lam_out": lam_out,
        "n_cg": ncg,
        "cg_converged": ncg < cg_max,
        "grad_norm": tnorm(g),
        "step_norm": tnorm(p),
        "finite": ok,
    }
    if plans is not None:
        info["block_min_eig"] = jnp.stack([jnp.min(jnp.linalg.eigvalsh(H)) for H in Hb])
        info["block_max_eig"] = jnp.stack([jnp.max(jnp.linalg.eigvalsh(H)) for H in Hb])
    return theta_out, r_out, lam_out, p_out, info


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
        label_fns = [mlp_unit_labels] * (len(bs) - 1) + [None]   # affine tail: no units
        _plans_cache[key] = [make_plan(b, m, lf, q_min) for b, lf in zip(bs, label_fns)]
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
    m=128 + 64,
    n_steps=60,
    lam0=1e-2,
    seed=0,
    verbose=True,
    precondition=True,
    min_loss=None,
):
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
    for i in range(n_steps):
        key, sk = jr.split(key)
        theta, r, lam, p_prev, info = step(
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
        )
        hist.append(info)
        if verbose:
            print(
                f"{i:3d}  F={float(info['F_new']):.4e}  "
                f"log(F)={float(np.log(info['F_new'])):+.2f} "
                f"rho={float(info['rho']):+.2f}  lam={float(info['lam_out']):.1e}  "
                f"cg={int(info['n_cg']):3d}{'' if info['cg_converged'] else '*'}  "
                f"|g|={float(info['grad_norm']):.2e}"
                f"{'' if info['accept'] else '   REJECT'}"
            )

        if not np.isfinite(info["grad_norm"]):
            if verbose:
                print("gradient norm is NaN or Inf; adding noise to parameters")
            theta = jax.tree.map(lambda x: 1e-2 * jr.normal(sk, x.shape), theta)

        if min_loss and info["F_new"] < min_loss:
            break
    return theta, hist
