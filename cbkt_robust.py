"""Additional simulations quoted in the paper (Method: the constant 1.5; Discussion A: bias estimated from
earlier responses; Limitations: weaker confidence-correctness link).

Differences from the main study (cbkt_study.py), and nothing else:
  * bias is a continuous learner trait, b_i ~ Normal(0, sigma_b), instead of three learner types;
  * "resolution" rho scales the answer-to-confidence link: the base level theta of Table I (panel C) is moved
    toward the midpoint, theta' = 0.5 + rho * (theta - 0.5)  (rho = 1 is Table I, rho = 0 leaves ratings that
    carry only bias and noise);
  * four ways of using the rating are compared with stdBKT, all with the same fitted P_G, P_S, P_T, P_L0:
      raw     cBKT as in the paper (CF = rating / 5)
      A       CF minus a bias estimate from the learner's EARLIER responses (shrinkage weight K_SHRINK)
      C       CF centred on the learner's own running mean rating
      oracle  CF minus the true bias (an upper bound, not available in practice)
  * the constant c in m = c - CF is varied.

Usage:
  python cbkt_robust.py                 # about 6 minutes: simulate, write results/robust.json and robust_summary.json
  python cbkt_robust.py --reps 3        # quick look (not comparable with the paper)
  python cbkt_robust.py --summarize     # only rebuild robust_summary.json from robust.json
"""
import argparse, json
import numpy as np
import cbkt_sim as s

T = s.T_OPP
K_SHRINK = 3.0                                   # shrinkage weight of option A (pseudo-observations)
SIGMAS = (0.2, 0.4)                              # close to the moderate and severe conditions
RESOLUTIONS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.0)
CONSTANTS = (1.0, 1.25, 1.5, 1.75, 2.0)
CONST_RESOLUTIONS = (1.0, 0.5)                   # resolutions at which the constant is varied
MODELS = ('raw', 'A', 'C', 'oracle')
SEED0 = 1000                                     # seeds 1000-1019, the paper's 20 new groups


# ---------------------------------------------------------------- ground truth (Table I, bias continuous)
def generate_cont(seed, sigma_b, rho=1.0):
    """Same hidden process and rating rule as cbkt_sim.generate (Table I), drawn for all learners at once.
    Learners are exchangeable, so the first 2,100 are the training learners and the last 900 the test learners
    (no permutation, unlike cbkt_sim.generate). The random stream therefore differs from the main study's."""
    N = s.N
    rng = np.random.default_rng(seed)
    pl0 = rng.uniform(0.1, 0.5, N)
    state = np.zeros((N, T), dtype=bool)
    state[:, 0] = rng.random(N) < pl0
    for t in range(1, T):
        state[:, t] = state[:, t - 1] | (rng.random(N) < s.PT_T)
    correct = np.where(state, rng.random((N, T)) > s.PS_T, rng.random((N, T)) < s.PG_T)
    confident_misconception = rng.random((N, T)) < 0.40
    theta = np.where(state & correct, 0.85,
            np.where(~state & correct, 0.45,
            np.where(state & ~correct, 0.35,
            np.where(confident_misconception, 0.80, 0.15))))
    bias = rng.normal(0.0, sigma_b, N)
    theta = 0.5 + rho * (theta - 0.5)                        # resolution: move theta toward the midpoint
    level = np.clip(theta + bias[:, None] + rng.normal(0, 0.08, (N, T)), 0.001, 0.999)
    rating = np.digitize(level, [0.2, 0.4, 0.6, 0.8]) + 1
    return (correct.astype(int), rating.astype(int), state.astype(int), bias,
            np.arange(2100), np.arange(2100, N))


# ---------------------------------------------------------------- ways of using the rating
def correct_history(C, cf, mu_right, mu_wrong, k=K_SHRINK):
    """Option A. Before opportunity t, the bias estimate is the sum of the learner's earlier deviations
    (rating minus the training-population mean rating for a right or wrong answer) divided by (n_earlier + k).
    It uses responses before t only and is 0 at t = 1."""
    d = cf - np.where(C == 1, mu_right, mu_wrong)
    earlier = np.cumsum(d, 1) - d
    b_hat = earlier / (np.arange(cf.shape[1])[None, :] + k)
    return np.clip(cf - b_hat, 0.0, 1.0)

def centre_on_own_mean(cf, cf_bar):
    """Option C. Re-centre each rating on the learner's own running mean (including the current rating)."""
    run = np.cumsum(cf, 1) / np.arange(1, cf.shape[1] + 1)[None, :]
    return np.clip(cf - run + cf_bar, 0.0, 1.0)

def at_under(over, under, tgt):
    o = np.argsort(under, kind='stable'); u = under[o]          # stable: identical in Python and JavaScript
    if tgt < u.min() or tgt > u.max(): return float('nan')
    return float(np.interp(tgt, u, over[o]))


def one_cont(seed, sigma_b, rho, constants=()):
    C, R, TL, bias, tr, te = generate_cont(seed, sigma_b, rho)
    PG, PS, PT, PL0 = s.fit_em(C[tr])
    cf_tr, ok_tr = R[tr] / 5.0, C[tr] == 1
    mu_right, mu_wrong, cf_bar = cf_tr[ok_tr].mean(), cf_tr[~ok_tr].mean(), cf_tr.mean()   # training learners only
    Cte, TLte, cf = C[te], TL[te], R[te] / 5.0
    inputs = {'stdBKT': None, 'raw': cf,
              'A': correct_history(Cte, cf, mu_right, mu_wrong),
              'C': centre_on_own_mean(cf, cf_bar),
              'oracle': np.clip(cf - bias[te][:, None], 0.0, 1.0)}
    def score(post):
        o, u = s.curves(post, TLte)
        return dict(over5=at_under(o, u, 0.05), mae=float(np.abs(post - TLte).mean()))
    out = {nm: score(s.run(Cte, x, PG, PS, PT, PL0)) for nm, x in inputs.items()}
    out['const'] = {f'{c:.2f}': score(s.run(Cte, cf, PG, PS, PT, PL0, c_off=c)) for c in constants}
    out['em'] = [PG, PS, PT, PL0]
    return out


# ---------------------------------------------------------------- summary
def summarize(R):
    def cell(rows, get):
        std = np.array([r['stdBKT']['over5'] for r in rows], float)
        v = np.array([get(r)['over5'] for r in rows], float)
        ok = ~np.isnan(v) & ~np.isnan(std)
        return dict(over=float(np.nanmean(v)), pct=float(np.mean((v[ok] - std[ok]) / std[ok]) * 100),
                    wins=int((v[ok] < std[ok]).sum()), reach=int((~np.isnan(v)).sum()),
                    mae=float(np.mean([get(r)['mae'] for r in rows])))
    out = {'res': {}, 'const': {}}
    for key, rows in R['cells'].items():
        out['res'][key] = {'std': dict(over=float(np.nanmean([r['stdBKT']['over5'] for r in rows])),
                                       mae=float(np.mean([r['stdBKT']['mae'] for r in rows])))}
        for m in MODELS: out['res'][key][m] = cell(rows, lambda r, m=m: r[m])
        if rows[0]['const']:
            for c in rows[0]['const']:
                out['const'][f'{key}|{c}'] = cell(rows, lambda r, c=c: r['const'][c])
    return out

def facts_robust(S):
    """The numbers quoted in the paper's additional-simulation sentences, as printed there."""
    r, k = S['res'], S['const']
    red = lambda sig, rho, m='raw': -r[f'{sig:.2f}|{rho:.2f}'][m]['pct']
    kr = lambda sig, cs: [-k[f'{sig:.2f}|1.00|{c:.2f}']['pct'] for c in cs]
    span = lambda v: f'{min(v):.0f}% to {max(v):.0f}%'
    return {
     "text_const_range_moderate": span(kr(0.2, (1.25, 1.5, 1.75))),
     "text_const_range_severe": span(kr(0.4, (1.25, 1.5, 1.75))),
     "text_const1_fails": f"at most {max(k[f'{x:.2f}|1.00|1.00']['reach'] for x in SIGMAS)} of 20 groups reach the 5% target",
     "text_history_vs_raw": (f"A {red(0.2,1,'A'):.0f}% vs raw {red(0.2,1):.0f}% (moderate); "
                             f"A {red(0.4,1,'A'):.0f}% vs raw {red(0.4,1):.0f}% (severe)"),
     "text_oracle_extra_pp_severe": f"{red(0.4,1,'oracle') - red(0.4,1):.0f}",
     "text_half_link": f"{red(0.2,0.5):.0f}% (moderate); {red(0.4,0.5):.0f}% (severe)",
     "text_no_information": f"{-red(0.2,0):.0f}% to {-red(0.4,0):.0f}% more practice"}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='results/robust.json'); ap.add_argument('--summary', default='results/robust_summary.json')
    ap.add_argument('--summarize', action='store_true'); a = ap.parse_args()
    if not a.summarize:
        R = {'cells': {}}
        for sig in SIGMAS:
            for rho in RESOLUTIONS:
                cs = CONSTANTS if rho in CONST_RESOLUTIONS else ()
                R['cells'][f'{sig:.2f}|{rho:.2f}'] = [one_cont(SEED0 + g, sig, rho, cs) for g in range(a.reps)]
                print(f'sigma_b={sig} rho={rho} done', flush=True)
        json.dump(R, open(a.out, 'w'))
    S = summarize(json.load(open(a.out)))
    json.dump(S, open(a.summary, 'w'))
    print('wrote', a.summary)
    for sig in SIGMAS:
        print(f'\nsigma_b = {sig}: change in over-practice vs stdBKT at 5% under-practice (mean of groups; groups better)')
        for rho in RESOLUTIONS:
            c = S['res'][f'{sig:.2f}|{rho:.2f}']
            print(f'  rho {rho:4.2f}  ' + '   '.join(f"{m} {c[m]['pct']:6.1f}% ({c[m]['wins']:2d})" for m in MODELS))
