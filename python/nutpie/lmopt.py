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
    return [np.broadcast_to(np.arange(W)[:, None], l1.weight.shape),   # w1 row
            np.arange(W),                                              # b1
            np.broadcast_to(np.arange(W)[None, :], l2.weight.shape),   # w2 col
            np.arange(l2.bias.shape[0]) % W]                           # b2 spread


def _perm_from_labels(block, label_fn):
    arr, _ = eqx.partition(block, eqx.is_inexact_array)
    one = jax.tree.map(lambda l: l[0], arr)
    leaves = jax.tree.leaves(one)
    labels = label_fn(one)
    assert [tuple(l.shape) for l in leaves] == [tuple(np.shape(g)) for g in labels], \
        "label shapes must match jax.tree.leaves order of the block"
    lab = np.concatenate([np.asarray(g).ravel() for g in labels])
    return np.argsort(lab, kind="stable")


def make_plan(block, m, label_fn=None, q_min=16):
    """Sub-block plan for one bucket.  q ~ m/2, capped at Pb, so small
    conditioners stay unsplit (a single group covering the whole block)."""
    flatten, unflatten, Pb = codec(block)
    perm = (_perm_from_labels(block, label_fn)
            if (label_fn is not None and Pb > 0) else np.arange(Pb))
    q = int(min(Pb, max(q_min, m // 2))) if Pb else 1
    G = -(-Pb // q) if Pb else 1
    idx = np.full(G * q, Pb, dtype=int)
    idx[:Pb] = perm                                   # column Pb = dummy sink
    return dict(flatten=flatten, unflatten=unflatten, Pb=Pb, q=q, G=G,
                idx=jnp.asarray(idx.reshape(G, q)),
                mask=jnp.asarray((idx < Pb).reshape(G, q)).astype(float))


def _gather(A, p):
    """(..., n, Pb) -> (..., n, G, q), padded entries zeroed."""
    A = jnp.concatenate([A, jnp.zeros(A.shape[:-1] + (1,), A.dtype)], -1)
    return A[..., p["idx"]] * p["mask"]


def _scatter(S, p):
    """(n, G, q) -> (n, Pb).  Live columns appear exactly once, so add == set."""
    n = S.shape[0]
    flat = jnp.zeros((n, p["Pb"] + 1), S.dtype).at[:, p["idx"]].add(S * p["mask"])
    return flat[:, :p["Pb"]]


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

    x, r, _, _, _, k = jax.lax.while_loop(
        cond, body, (x0, r0, z0, z0, tdot(r0, z0), 0))
    return x, k


# ============================================================ build and apply

def make_block_builder(res_fn, plans, m, batch=32):
    """Returns build(params, key) -> [ (n_j, G_j, q_j, q_j) ] per bucket.

    E[(G^T w)(G^T w)^T] = G^T G for isotropic w, so a single VJP updates every
    block at once and the estimate is unbiased and PSD by construction.
    """
    def build(params, key):
        _, vjpf = jax.vjp(res_fn, params)
        shape = jax.eval_shape(res_fn, params).shape            # (N, d)

        def one(k):
            w = jr.rademacher(k, shape).astype(jnp.float64)
            g = vjpf(w)[0]
            return [_gather(p["flatten"](b), p)
                    for b, p in zip(blocks_of(g), plans)]

        Gs = jax.lax.map(one, jr.split(key, m), batch_size=batch)
        return [jnp.einsum('tngi,tngj->ngij', G, G) / m for G in Gs]

    return build


def make_apply(blocks, params, plans, lam):
    """Returns Minv(v) -> pytree.  eigh with clamping rather than Cholesky:
    blocks are small, some are structurally singular (a conditioner with
    in_features=0 has a draw-independent Jacobian), and eigh avoids the static
    `lower` flag that breaks under vmap."""
    def inv(H):
        w, V = jnp.linalg.eigh(H)
        return jnp.einsum('ij,j,kj->ik', V, 1.0 / (jnp.maximum(w, 0.0) + lam), V)

    Minvs = [jax.vmap(jax.vmap(inv))(H) for H in blocks]        # (n, G, q, q)

    def Minv(v):
        out = []
        for b, p, Mi in zip(blocks_of(v), plans, Minvs):
            S = _gather(p["flatten"](b), p)                     # (n, G, q)
            S = jnp.einsum('ngij,ngj->ngi', Mi, S)
            out.append(p["unflatten"](_scatter(S, p)))
        return rebuild(v, out)

    return Minv


# ============================================================ LM step

def lm_step(res_fn, args, build_blocks, make_apply_fn, theta, r, lam, key,
            cg_tol=1e-2, cg_max=300, p_prev=None,
            accept_rho=0.1, good_rho=0.75,
            lam_down=3.0, lam_up=4.0, lam_min=1e-10, lam_max=1e10):
    """One LM step.  The three callables are static under eqx.filter_jit."""
    res_fn_args = lambda p: res_fn(p, args)
    Hb = build_blocks(theta, key)
    Minv = make_apply_fn(Hb, theta, lam)

    jvp = lambda v: jax.jvp(res_fn_args, (theta,), (v,))[1]
    _, vjpf = jax.vjp(res_fn_args, theta)
    vjp = lambda w: vjpf(w)[0]
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
    pred = -tdot(p, g) - 0.5 * tdot(Jp, Jp)          # undamped GN model
    rho = actual / jnp.where(jnp.abs(pred) < 1e-30, 1e-30, pred)

    ok = jnp.isfinite(F_new) & jnp.isfinite(rho)
    accept = (rho > accept_rho) & ok

    pick = lambda a, b: jax.tree.map(lambda x, y: jnp.where(accept, x, y), a, b)
    theta_out, r_out = pick(theta_new, theta), pick(r_new, r)
    lam_out = jnp.clip(
        jnp.where(accept, jnp.where(rho > good_rho, lam / lam_down, lam),
                  lam * lam_up), lam_min, lam_max)
    # a rejected step's p solves a system with a different lambda: discard it
    p_out = jax.tree.map(lambda a: jnp.where(accept, a, jnp.zeros_like(a)), p)

    info = dict(
        F=F, F_new=F_new, accept=accept, rho=rho, actual=actual, pred=pred,
        lam_in=lam, lam_out=lam_out, n_cg=ncg, cg_converged=ncg < cg_max,
        grad_norm=tnorm(g), step_norm=tnorm(p), finite=ok,
        block_min_eig=jnp.stack([jnp.min(jnp.linalg.eigvalsh(H)) for H in Hb]),
        block_max_eig=jnp.stack([jnp.max(jnp.linalg.eigvalsh(H)) for H in Hb]),
    )
    return theta_out, r_out, lam_out, p_out, info


step = eqx.filter_jit(lm_step)


# ============================================================ setup and driver

def setup(params, res_fn, args, m=128, q_min=16):
    """Build plans and the two closures lm_step expects."""
    bs = blocks_of(params)
    label_fns = [mlp_unit_labels] * (len(bs) - 1) + [None]   # affine tail: no units
    plans = [make_plan(b, m, lf, q_min) for b, lf in zip(bs, label_fns)]

    assert jax.tree.all(jax.tree.map(
        lambda a, b: bool(jnp.all(a == b)),
        rebuild(params, blocks_of(params)), params)), \
        "blocks_of / rebuild do not round-trip"

    build = make_block_builder(lambda p: res_fn(p, args), plans, m)
    apply_fn = lambda Hb, th, lam: make_apply(Hb, th, plans, lam)
    return build, apply_fn, plans


def fit(params, res_fn, args, m=128, n_steps=60, lam0=1e-2, seed=0, verbose=True, precondition=True, min_loss=None):
    build, apply_fn, plans = setup(params, res_fn, args)
    if not precondition:
        apply_fn = lambda Hb, th, lam: (lambda v: v)
        build = lambda th, key: [jnp.zeros((1, 1, 1, 1))] * len(plans)

    if verbose:
        print("bucket (Pb, q, n_subblocks):",
              [(p["Pb"], p["q"], p["G"]) for p in plans])

    theta, r, lam = params, res_fn(params, args), jnp.asarray(lam0)
    p_prev, key, hist = None, jr.key(seed), []
    for i in range(n_steps):
        key, sk = jr.split(key)
        theta, r, lam, p_prev, info = step(
            res_fn, args, build, apply_fn, theta, r, lam, sk, p_prev=p_prev)
        hist.append(info)
        if verbose:
            print(f"{i:3d}  F={float(info['F_new']):.4e}  "
                  f"log(F)={float(np.log(info['F_new'])):+.2f} "
                  f"rho={float(info['rho']):+.2f}  lam={float(info['lam_out']):.1e}  "
                  f"cg={int(info['n_cg']):3d}{'' if info['cg_converged'] else '*'}  "
                  f"|g|={float(info['grad_norm']):.2e}"
                  f"{'' if info['accept'] else '   REJECT'}")
        if min_loss and info['F_new'] < min_loss:
            break
    return theta, hist
