"""Analyses added for the camera-ready revision (AIESC 2027, paper C1142), one per reviewer comment.

  Comment 1  rating noise SD varied (0.08 = paper, 0.15, 0.25, 0.35) in the main three-type generator,
             with the resolution of the ratings (gamma and AUROC of rating vs correctness) for each setting.
  Comment 2  (a) the rule against the data: P(correct | state, rating) in the generator vs the guess and slip
                 that cBKT uses for each rating;
             (b) where the gain comes from: confidence applied in one response cell only (rating band x answer);
             (c) asymmetric variants: guess-only (slip fixed at P_S) and correct-answers-only.
  Comment 3  feature-based model refitted without the 0.40 cap, with a convergence check, 5 EM restarts
             (best training log-likelihood) and the tau grid extended above 0.99; also the training-size runs.
             The stdBKT EM is checked for convergence at 80 iterations.

Everything else is as in cbkt_sim.py / cbkt_study.py. At noise SD 0.08 the generator below draws exactly the same
random stream as cbkt_sim.generate, so the paper's main numbers are reproduced (checked at start-up).
Usage: python cbkt_revision.py [--reps 20] [--out results/revision.json]   (about 9 minutes)
       python summarize_revision.py                                          (summary + printed tables)
"""
import argparse, json, time
import numpy as np
import cbkt_sim as s
import cbkt_study as st

T = s.T_OPP
NOISE = (0.08, 0.15, 0.25, 0.35)
TAUS_EXT = np.concatenate([s.TAUS, np.linspace(0.991, 0.999, 9), [0.9995, 0.9999]])


# ------------------------------------------------------------------ generator with a noise parameter
def generate(seed, mix, bias, sd=0.08):
    """cbkt_sim.generate with the rating-noise SD as a parameter (identical random stream for every sd)."""
    N = s.N
    rng = np.random.default_rng(seed)
    pl0 = rng.uniform(0.1, 0.5, N)
    types = rng.choice(3, size=N, p=mix)
    C = np.zeros((N, T), int); R = np.zeros_like(C); TL = np.zeros_like(C)
    for i in range(N):
        L = int(rng.random() < pl0[i])
        for t in range(T):
            c = int(rng.random() < ((1 - s.PS_T) if L else s.PG_T))
            if c and L: th = 0.85
            elif c and not L: th = 0.45
            elif (not c) and L: th = 0.35
            else: th = 0.15 if rng.random() < 0.6 else 0.80
            if types[i] == 1: th += bias
            elif types[i] == 2: th -= bias
            th = np.clip(th + rng.normal(0, sd), 0.001, 0.999)
            R[i, t] = int(np.digitize(th, [0.2, 0.4, 0.6, 0.8])) + 1
            C[i, t] = c; TL[i, t] = L
            if L == 0 and rng.random() < s.PT_T: L = 1
    idx = rng.permutation(N)
    return C, R, TL, types, idx[:2100], idx[2100:]


# ------------------------------------------------------------------ resolution measures
def gamma(r, c):
    """Goodman-Kruskal gamma between rating (1-5) and correctness (0/1)."""
    n1 = np.array([((r == k) & (c == 1)).sum() for k in range(1, 6)], float)
    n0 = np.array([((r == k) & (c == 0)).sum() for k in range(1, 6)], float)
    conc = sum(n0[i] * n1[i + 1:].sum() for i in range(5))
    disc = sum(n1[i] * n0[i + 1:].sum() for i in range(5))
    return float((conc - disc) / (conc + disc))

def _auc(r, c):
    """AUROC: P(rating of a random correct answer > that of a random wrong answer), ties counted 1/2."""
    n1 = np.array([((r == k) & (c == 1)).sum() for k in range(1, 6)], float)
    n0 = np.array([((r == k) & (c == 0)).sum() for k in range(1, 6)], float)
    gt = sum(n1[i] * n0[:i].sum() for i in range(5)); eq = (n1 * n0).sum()
    return float((gt + 0.5 * eq) / (n1.sum() * n0.sum()))


# ------------------------------------------------------------------ tracking variants
def run_variant(C, R, PG, PS, PT, PL0, use_g=True, use_s=True, cell=None):
    """cBKT update with confidence used only where allowed.
    use_g / use_s: adjust guess / slip with m = 1.5 - CF (else keep P_G / P_S).
    cell: (band, answer) -> confidence used only for responses in that rating band and answer
          (band 'low' = 1-2, 'mid' = 3, 'high' = 4-5; answer 1 = correct, 0 = wrong); elsewhere stdBKT."""
    n = C.shape[0]; L = np.full(n, PL0); post = np.zeros((n, T))
    bands = {'low': (1, 2), 'mid': (3, 3), 'high': (4, 5)}
    for t in range(T):
        m = 1.5 - R[:, t] / 5.0
        if cell is not None:
            lo, hi = bands[cell[0]]
            on = (R[:, t] >= lo) & (R[:, t] <= hi) & (C[:, t] == cell[1])
            m = np.where(on, m, 1.0)
        pg = np.clip(PG * m, s.CLIP_LO, s.CLIP_HI) if use_g else np.full(n, PG)
        ps = np.clip(PS * m, s.CLIP_LO, s.CLIP_HI) if use_s else np.full(n, PS)
        if cell is not None:                      # outside the cell: exactly stdBKT (no clip effect)
            pg = np.where(m == 1.0, PG, pg); ps = np.where(m == 1.0, PS, ps)
        p = L * (1 - ps) + (1 - L) * pg; c = C[:, t]
        q = np.where(c == 1, L * (1 - ps) / p, L * ps / (1 - p))
        post[:, t] = q; L = q + (1 - q) * PT
    return post


# ------------------------------------------------------------------ fair feature-based model
def loglik(E, PT, PL0):
    A = np.array([[1 - PT, PT], [0, 1.0]]); a = np.array([1 - PL0, PL0]) * E[:, 0]
    ll = np.log(a.sum(1)); a /= a.sum(1, keepdims=True)
    for t in range(1, E.shape[1]):
        a = E[:, t] * (a @ A); z = a.sum(1); ll += np.log(z); a /= z[:, None]
    return float(ll.sum())

def emis(C, R, pc):
    E = np.zeros(C.shape + (2,))
    for st_ in (0, 1):
        p = pc[st_][R - 1]; E[..., st_] = np.where(C == 1, p, 1 - p)
    return E

def fit_fast_fair(C, R, restarts=5, max_iter=1000, tol=1e-6, seed=0, lo=0.001, hi=0.999):
    """P(correct | state, rating) as ten free values, bounded only to [0.001, 0.999]; EM until the largest
    parameter change is below tol (or max_iter); best of `restarts` starts by training log-likelihood.
    Labels are fixed afterwards so that state 1 is the state with the higher mean P(correct)."""
    rng = np.random.default_rng(seed); best = None
    for k in range(restarts):
        if k == 0: pc = np.zeros((2, 5)); pc[0, :] = .25; pc[1, :] = .85; PT, PL0 = .10, .30
        else:
            pc = np.vstack([rng.uniform(.05, .5, 5), rng.uniform(.5, .95, 5)])
            PT, PL0 = rng.uniform(.05, .3), rng.uniform(.1, .5)
        for it in range(max_iter):
            g, xi = s.fb(emis(C, R, pc), PT, PL0)
            new = pc.copy()
            for st_ in (0, 1):
                for j in range(5):
                    w = g[..., st_] * (R == j + 1)
                    new[st_, j] = (w * C).sum() / max(w.sum(), 1e-9) if w.sum() > 1e-9 else pc[st_, j]
            new = np.clip(new, lo, hi)
            nPT = np.clip(xi[:, :, 0, 1].sum() / g[..., 0][:, :-1].sum(), .001, .6)
            nPL0 = np.clip(g[:, 0, 1].mean(), .001, .999)
            delta = max(np.abs(new - pc).max(), abs(nPT - PT), abs(nPL0 - PL0))
            pc, PT, PL0 = new, nPT, nPL0
            if delta < tol: break
        # Label switch: swap the emission rows if state 1 is the less accurate one. P_T and P_L0 are not swapped,
        # so this is only a safeguard; it never triggered in the reported runs (checked for noise SD 0.08 and 0.35).
        if pc[1].mean() < pc[0].mean():
            pc = pc[::-1].copy()
        ll = loglik(emis(C, R, pc), PT, PL0)
        if best is None or ll > best[0]: best = (ll, pc.copy(), PT, PL0, it + 1)
    return best[1], best[2], best[3], dict(ll=best[0], iters=best[4])

def fit_em_conv(C, max_iter=2000, tol=1e-8):
    """stdBKT EM as in cbkt_sim.fit_em but run to convergence (for the convergence check only)."""
    PG, PS, PT, PL0 = .25, .15, .10, .30
    for it in range(max_iter):
        E = np.zeros(C.shape + (2,))
        E[..., 1] = np.where(C == 1, 1 - PS, PS); E[..., 0] = np.where(C == 1, PG, 1 - PG)
        g, xi = s.fb(E, PT, PL0); g0, g1 = g[..., 0], g[..., 1]; ok = C == 1
        n = (np.clip((g0 * ok).sum() / g0.sum(), .05, .4), np.clip((g1 * ~ok).sum() / g1.sum(), .05, .3),
             np.clip(xi[:, :, 0, 1].sum() / g0[:, :-1].sum(), .02, .4), np.clip(g[:, 0, 1].mean(), .05, .6))
        d = max(abs(a - b) for a, b in zip(n, (PG, PS, PT, PL0))); PG, PS, PT, PL0 = n
        if d < tol: break
    return [float(x) for x in (PG, PS, PT, PL0)], it + 1


# ------------------------------------------------------------------ scoring
def curves(post, TL, taus):
    ok = TL.any(1); m = np.where(ok, TL.argmax(1) + 1, 0)
    ge = post[:, :, None] >= taus[None, None, :]
    declared = ge.any(1); first = np.where(declared, ge.argmax(1) + 1, T)
    over = (np.maximum(first - m[:, None], 0) * ok[:, None]).sum(0) / ok.sum()
    early = declared & ok[:, None] & (first < m[:, None]); never = declared & ~ok[:, None]
    return over, (early | never).sum(0) / len(post)

def score(post, TL, taus=s.TAUS):
    o, u = curves(post, TL, taus)
    return dict(over5=st.at_under(o, u, .05), thr5=_thr(u, taus, .05), mae=float(np.abs(post - TL).mean()))

def _thr(u, taus, tgt):
    o = np.argsort(u); uu = u[o]
    if tgt < uu.min() or tgt > uu.max(): return float('nan')
    return float(np.interp(tgt, uu, taus[o]))

CELLS = [(b, a) for a in (1, 0) for b in ('low', 'mid', 'high')]

def one(seed, scen, sd, fair_fb=False, train_n=None, diag=False):
    mix, bias = s.SCEN[scen]
    C, R, TL, ty, tr, te = generate(seed, mix, bias, sd)
    if train_n: tr = tr[:train_n]
    PG, PS, PT, PL0 = s.fit_em(C[tr])
    Cte, Rte, TLte = C[te], R[te], TL[te]
    out = {'em': [PG, PS, PT, PL0],
           'gamma': gamma(R[te], C[te]), 'auroc': _auc(R[te], C[te]),
           'gamma_state': gamma(R[te], TL[te])}
    posts = {'stdBKT': s.run(Cte, None, PG, PS, PT, PL0), 'cBKT': s.run(Cte, Rte / 5.0, PG, PS, PT, PL0)}
    if train_n is None:
        posts['g_only'] = run_variant(Cte, Rte, PG, PS, PT, PL0, use_s=False)
        posts['correct_only'] = _correct_only(Cte, Rte, PG, PS, PT, PL0)
        for cell in CELLS:
            posts['cell_%s_%d' % cell] = run_variant(Cte, Rte, PG, PS, PT, PL0, cell=cell)
    for nm, p in posts.items(): out[nm] = score(p, TLte)
    if fair_fb:
        pc0, fPT0, fPL00 = st.fit_fast(C[tr], R[tr])                       # paper's version
        out['FB_paper'] = score(st.run_fast(Cte, Rte, pc0, fPT0, fPL00), TLte)
        out['FB_paper_ext'] = score(st.run_fast(Cte, Rte, pc0, fPT0, fPL00), TLte, TAUS_EXT)
        pc, fPT, fPL0, info = fit_fast_fair(C[tr], R[tr], seed=seed)
        pf = st.run_fast(Cte, Rte, pc, fPT, fPL0)
        out['FB_fair'] = score(pf, TLte, TAUS_EXT)
        out['FB_fair']['pc'] = pc.tolist(); out['FB_fair'].update(info)
        out['cBKT_ext'] = score(posts['cBKT'], TLte, TAUS_EXT)
        out['stdBKT_ext'] = score(posts['stdBKT'], TLte, TAUS_EXT)
    if diag:   # P(correct | state, rating) in the training data (the "true" guess/slip per rating)
        Ctr, Rtr, Str = C[tr], R[tr], TL[tr]
        tab = {}
        for k in range(1, 6):
            nm_ = (Rtr == k) & (Str == 0); ms_ = (Rtr == k) & (Str == 1)
            tab[k] = dict(g_true=float(Ctr[nm_].mean()) if nm_.any() else None,
                          s_true=float(1 - Ctr[ms_].mean()) if ms_.any() else None,
                          n_nm=int(nm_.sum()), n_m=int(ms_.sum()),
                          g_cbkt=float(np.clip(PG * (1.5 - k / 5), .05, .4)),
                          s_cbkt=float(np.clip(PS * (1.5 - k / 5), .05, .4)),
                          share_correct=float(((Rtr == k) & (Ctr == 1)).mean()),
                          share_wrong=float(((Rtr == k) & (Ctr == 0)).mean()),
                          p_nm_given_wrong=float((Str[(Rtr == k) & (Ctr == 0)] == 0).mean()) if ((Rtr == k) & (Ctr == 0)).any() else None,
                          p_nm_given_right=float((Str[(Rtr == k) & (Ctr == 1)] == 0).mean()) if ((Rtr == k) & (Ctr == 1)).any() else None)
        out['diag'] = tab
        out['em_conv'], out['em_conv_iters'] = fit_em_conv(C[tr])
    return out

def _correct_only(C, R, PG, PS, PT, PL0):
    """Confidence used only after correct answers; wrong answers get the stdBKT update."""
    n = C.shape[0]; L = np.full(n, PL0); post = np.zeros((n, T))
    for t in range(T):
        m = 1.5 - R[:, t] / 5.0; c = C[:, t]
        pg = np.where(c == 1, np.clip(PG * m, s.CLIP_LO, s.CLIP_HI), PG)
        ps = np.where(c == 1, np.clip(PS * m, s.CLIP_LO, s.CLIP_HI), PS)
        p = L * (1 - ps) + (1 - L) * pg
        q = np.where(c == 1, L * (1 - ps) / p, L * ps / (1 - p))
        post[:, t] = q; L = q + (1 - q) * PT
    return post


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='results/revision.json'); a = ap.parse_args()
    # sanity: at sd 0.08 the generator reproduces cbkt_sim.generate exactly
    x = generate(42, *s.SCEN['moderate'], 0.08); y = s.generate(42, *s.SCEN['moderate'])
    assert all(np.array_equal(p, q) for p, q in zip(x, y)), 'generator differs from the paper'
    print('generator check ok', flush=True)
    R = {}
    t0 = time.time()
    for scen in s.SCEN:
        R[scen] = {}
        for sd in NOISE:
            R[scen][f'{sd:.2f}'] = [one(1000 + g, scen, sd, fair_fb=True, diag=(g == 0)) for g in range(a.reps)]
            print(scen, sd, 'done %.0fs' % (time.time() - t0), flush=True)
        R[scen]['sizes'] = {n_: [one(1000 + g, scen, 0.08, fair_fb=True, train_n=n_) for g in range(a.reps)]
                            for n_ in (10, 50, 200)}
        print(scen, 'sizes done %.0fs' % (time.time() - t0), flush=True)
    json.dump(R, open(a.out, 'w'))
    print('wrote', a.out)
