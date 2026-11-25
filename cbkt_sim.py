"""Replication script: stdBKT vs cBKT in the hidden-state simulation (see paper, Sec. Simulation Design).
Usage: python cbkt_sim.py            # quick check: prints the 5%-under-practice numbers for seed 42
"""
import numpy as np
T_OPP, N = 15, 3000
PG_T, PS_T, PT_T = 0.30, 0.20, 0.15
CLIP_LO, CLIP_HI = 0.05, 0.40
TAUS = np.linspace(0.50, 0.99, 40)
SCEN = {"moderate": ([0.5, 0.25, 0.25], 0.25), "severe": ([0.3, 0.35, 0.35], 0.40)}

def generate(seed, mix, bias):
    rng = np.random.default_rng(seed)
    pl0 = rng.uniform(0.1, 0.5, N)
    types = rng.choice(3, size=N, p=mix)            # 0 calibrated, 1 over, 2 under
    C = np.zeros((N, T_OPP), int); R = np.zeros_like(C); TL = np.zeros_like(C)
    for i in range(N):
        L = int(rng.random() < pl0[i])
        for t in range(T_OPP):
            c = int(rng.random() < ((1 - PS_T) if L else PG_T))
            if c and L: th = 0.85
            elif c and not L: th = 0.45
            elif (not c) and L: th = 0.35
            else: th = 0.15 if rng.random() < 0.6 else 0.80
            if types[i] == 1: th += bias
            elif types[i] == 2: th -= bias
            th = np.clip(th + rng.normal(0, 0.08), 0.001, 0.999)
            R[i, t] = int(np.digitize(th, [0.2, 0.4, 0.6, 0.8])) + 1
            C[i, t] = c; TL[i, t] = L
            if L == 0 and rng.random() < PT_T: L = 1
    idx = rng.permutation(N)
    return C, R, TL, types, idx[:2100], idx[2100:]

def fb(E, PT, PL0):
    n, T, _ = E.shape
    A = np.array([[1 - PT, PT], [0, 1.0]]); pi = np.array([1 - PL0, PL0])
    al = np.zeros((n, T, 2)); al[:, 0] = pi * E[:, 0]
    for t in range(1, T): al[:, t] = E[:, t] * (al[:, t - 1] @ A)
    be = np.ones((n, T, 2))
    for t in range(T - 2, -1, -1): be[:, t] = (E[:, t + 1] * be[:, t + 1]) @ A.T
    g = al * be; g /= np.clip(g.sum(2, keepdims=True), 1e-300, None)
    xi = np.einsum('ntl,lm,ntm->ntlm', al[:, :-1], A, E[:, 1:] * be[:, 1:])
    xi /= np.clip(xi.sum((2, 3), keepdims=True), 1e-300, None)
    return g, xi

def fit_em(C, iters=80):
    PG, PS, PT, PL0 = .25, .15, .10, .30
    for _ in range(iters):
        E = np.zeros(C.shape + (2,))
        E[..., 1] = np.where(C == 1, 1 - PS, PS); E[..., 0] = np.where(C == 1, PG, 1 - PG)
        g, xi = fb(E, PT, PL0); g0, g1 = g[..., 0], g[..., 1]; ok = C == 1
        PG = np.clip((g0 * ok).sum() / g0.sum(), .05, .4)
        PS = np.clip((g1 * ~ok).sum() / g1.sum(), .05, .3)
        PT = np.clip(xi[:, :, 0, 1].sum() / g0[:, :-1].sum(), .02, .4)
        PL0 = np.clip(g[:, 0, 1].mean(), .05, .6)
    return PG, PS, PT, PL0

def run(C, cf, PG, PS, PT, PL0, c_off=1.5):
    """c_off is the constant in the multiplier m = c_off - CF; the paper uses 1.5 (other values: cbkt_robust.py)."""
    n, T = C.shape; L = np.full(n, PL0); post = np.zeros((n, T))
    for t in range(T):
        if cf is None: pg, ps = PG, PS
        else:
            pg = np.clip(PG * (c_off - cf[:, t]), CLIP_LO, CLIP_HI)
            ps = np.clip(PS * (c_off - cf[:, t]), CLIP_LO, CLIP_HI)
        p = L * (1 - ps) + (1 - L) * pg; c = C[:, t]
        q = np.where(c == 1, L * (1 - ps) / p, L * ps / (1 - p))
        post[:, t] = q; L = q + (1 - q) * PT
    return post

def curves(post, TL):
    ok = TL.any(1); m = np.where(ok, TL.argmax(1) + 1, 0)
    ge = post[:, :, None] >= TAUS[None, None, :]
    declared = ge.any(1); first = np.where(declared, ge.argmax(1) + 1, T_OPP)
    over = (np.maximum(first - m[:, None], 0) * ok[:, None]).sum(0) / ok.sum()
    early = declared & ok[:, None] & (first < m[:, None]); never = declared & ~ok[:, None]
    under = (early | never).sum(0) / len(post)
    return over, under

def over_at(over, under, tgt):
    o = np.argsort(under); return float(np.interp(tgt, under[o], over[o]))

if __name__ == "__main__":
    for scen, (mix, bias) in SCEN.items():
        C, R, TL, ty, tr, te = generate(42, mix, bias)
        PG, PS, PT, PL0 = fit_em(C[tr])
        s = run(C[te], None, PG, PS, PT, PL0)
        c = run(C[te], R[te] / 5.0, PG, PS, PT, PL0)
        o1, u1 = curves(s, TL[te]); o2, u2 = curves(c, TL[te])
        print(scen, "EM", np.round([PG, PS, PT, PL0], 3),
              "over@5pct: std %.2f cBKT %.2f" % (over_at(o1, u1, .05), over_at(o2, u2, .05)))
