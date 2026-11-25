"""Full study: stdBKT vs cBKT vs feature-based (FAST-style) model. Produces every number in Tables II-IV.
Usage: python cbkt_study.py [--boot 2000] [--reps 20] [--out results/study.json]"""
import argparse, json, numpy as np
import cbkt_sim as s
T = s.T_OPP

# ---------------- feature-based model: P(correct | state, rating) learned by EM (10 values)
def fit_fast(C, R, iters=80):
    PT, PL0 = .10, .30
    pc = np.zeros((2, 5)); pc[0, :] = .25; pc[1, :] = .85          # P(correct | state, rating)
    for _ in range(iters):
        E = np.zeros(C.shape + (2,))
        for st in (0, 1):
            p = pc[st][R - 1]; E[..., st] = np.where(C == 1, p, 1 - p)
        g, xi = s.fb(E, PT, PL0)
        for st in (0, 1):
            for k in range(5):
                m = (R == k + 1); w = g[..., st] * m
                pc[st, k] = (w * C).sum() / max(w.sum(), 1e-9)
        pc[0] = np.clip(pc[0], 0.02, 0.40)                          # guess <= 0.40 (same ceiling as cBKT)
        pc[1] = np.clip(pc[1], 0.60, 0.98)                          # slip  <= 0.40
        PT = np.clip(xi[:, :, 0, 1].sum() / g[..., 0][:, :-1].sum(), .02, .4)
        PL0 = np.clip(g[:, 0, 1].mean(), .05, .6)
    return pc, PT, PL0

def run_fast(C, R, pc, PT, PL0):
    n = C.shape[0]; L = np.full(n, PL0); post = np.zeros((n, T))
    for t in range(T):
        p1 = pc[1][R[:, t] - 1]; p0 = pc[0][R[:, t] - 1]
        c = C[:, t]
        l1 = np.where(c == 1, p1, 1 - p1); l0 = np.where(c == 1, p0, 1 - p0)
        q = L * l1 / (L * l1 + (1 - L) * l0)
        post[:, t] = q; L = q + (1 - q) * PT
    return post

# ---------------- metrics
def under_over_curves(post, TL): return s.curves(post, TL)

def at_under(over, under, tgt):
    o = np.argsort(under); u = under[o]
    if tgt < u.min() or tgt > u.max(): return np.nan                # target not reachable
    return float(np.interp(tgt, u, over[o]))

def thr_at_under(under, tgt):
    o = np.argsort(under); u = under[o]
    if tgt < u.min() or tgt > u.max(): return np.nan
    return float(np.interp(tgt, u, s.TAUS[o]))

def under_at_over(over, under, tgt):
    o = np.argsort(over)
    if tgt < over.min() or tgt > over.max(): return np.nan
    return float(np.interp(tgt, over[o], under[o]))

def opps(post, thr=.85):
    ge = post >= thr; return np.where(ge.any(1), ge.argmax(1) + 1, T)

def one(seed, scen, train_n=None):
    mix, bias = s.SCEN[scen]
    C, R, TL, ty, tr, te = s.generate(seed, mix, bias)
    if train_n: tr = tr[:train_n]
    PG, PS, PT, PL0 = s.fit_em(C[tr])
    pc, fPT, fPL0 = fit_fast(C[tr], R[tr])
    Cte, Rte, TLte, tyte = C[te], R[te], TL[te], ty[te]
    posts = {'stdBKT': s.run(Cte, None, PG, PS, PT, PL0),
             'cBKT': s.run(Cte, Rte / 5.0, PG, PS, PT, PL0),
             'FB': run_fast(Cte, Rte, pc, fPT, fPL0)}
    res = {'em': [PG, PS, PT, PL0], 'types': [int((tyte == k).sum()) for k in range(3)]}
    ok = TLte.any(1); m = np.where(ok, TLte.argmax(1) + 1, T)
    res['true_opps'] = float(m.mean())
    res['m'] = {}
    for nm, p in posts.items():
        o, u = under_over_curves(p, TLte)
        err = np.abs(p - TLte)
        res[nm] = dict(over5=at_under(o, u, .05), over3=at_under(o, u, .03), under25=under_at_over(o, u, 2.5),
                       thr5=thr_at_under(u, .05), mae=float(err.mean()), opps=float(opps(p).mean()),
                       mae_type=[float(err[tyte == k].mean()) for k in range(3)], over=o.tolist(), under=u.tolist())
    return res, (posts, TLte, tyte, ok, m)

def boot(seed, scen, B):
    mix, bias = s.SCEN[scen]
    res, (posts, TLte, tyte, ok, m) = one(seed, scen)
    rng = np.random.default_rng(seed + 999); n = len(TLte)
    ge = {nm: p[:, :, None] >= s.TAUS[None, None, :] for nm, p in posts.items()}
    first = {nm: np.where(g.any(1), g.argmax(1) + 1, T) for nm, g in ge.items()}
    dec = {nm: g.any(1) for nm, g in ge.items()}
    op = {nm: opps(p) for nm, p in posts.items()}
    err = {nm: np.abs(p - TLte).mean(1) for nm, p in posts.items()}
    err_by = {nm: np.abs(p - TLte) for nm, p in posts.items()}
    out = {k: [] for k in ['over5', 'under25', 'mae', 'opps', 'mae0', 'mae1', 'mae2']}
    for _ in range(B):
        ids = rng.integers(0, n, n); w = np.bincount(ids, minlength=n).astype(float)
        cur = {}
        for nm in ('stdBKT', 'cBKT'):
            over = (np.maximum(first[nm] - m[:, None], 0) * ok[:, None] * w[:, None]).sum(0) / (ok * w).sum()
            early = dec[nm] & ok[:, None] & (first[nm] < m[:, None]); never = dec[nm] & ~ok[:, None]
            under = ((early | never) * w[:, None]).sum(0) / w.sum()
            cur[nm] = (at_under(over, under, .05), under_at_over(over, under, 2.5),
                       (err[nm] * w).sum() / w.sum(), (op[nm] * w).sum() / w.sum(),
                       [(err[nm] * w)[tyte == k].sum() / w[tyte == k].sum() for k in range(3)])
        out['over5'].append(cur['cBKT'][0] - cur['stdBKT'][0]); out['under25'].append(cur['cBKT'][1] - cur['stdBKT'][1])
        out['mae'].append(cur['cBKT'][2] - cur['stdBKT'][2]); out['opps'].append(cur['cBKT'][3] - cur['stdBKT'][3])
        for k in range(3): out['mae%d' % k].append(cur['cBKT'][4][k] - cur['stdBKT'][4][k])
    ci = {k: (float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}
    return res, ci

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='results/study.json'); a = ap.parse_args()
    allr = {}
    for scen in s.SCEN:
        main, ci = boot(42, scen, a.boot)
        reps = [one(1000 + g, scen)[0] for g in range(a.reps)]
        sizes = {}
        for n_ in (10, 50, 200):
            sizes[n_] = [one(1000 + g, scen, train_n=n_)[0] for g in range(a.reps)]
        allr[scen] = dict(main=main, ci=ci, reps=reps, sizes=sizes)
        print(scen, 'done', flush=True)
    json.dump(allr, open(a.out, 'w'))
