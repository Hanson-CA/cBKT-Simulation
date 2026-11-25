"""Summarize results/revision.json (written by cbkt_revision.py) into results/revision_summary.json and print it.
Change = mean over groups of (model - stdBKT) / stdBKT over-practice at 5% under-practice, groups where either
curve never reaches 5% left out (counted in `reach`)."""
import json, sys
import numpy as np

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/revision.json'))
CELLS = ['cell_%s_%d' % (b, a) for a in (1, 0) for b in ('low', 'mid', 'high')]

def cell(rows, nm, base='stdBKT'):
    v = np.array([r[nm]['over5'] for r in rows], float); b = np.array([r[base]['over5'] for r in rows], float)
    ok = ~np.isnan(v) & ~np.isnan(b)
    thr = np.array([r[nm].get('thr5', np.nan) for r in rows], float)
    return dict(over=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                pct=float(np.mean((v[ok] - b[ok]) / b[ok]) * 100) if ok.any() else None,
                wins=int((v[ok] < b[ok]).sum()), reach=int((~np.isnan(v)).sum()),
                thr=float(np.nanmean(thr)) if (~np.isnan(thr)).any() else None,
                mae=float(np.mean([r[nm]['mae'] for r in rows])))

out = {}
for scen, d in R.items():
    out[scen] = {}
    for sd, rows in d.items():
        if sd == 'sizes': continue
        o = dict(gamma=float(np.mean([r['gamma'] for r in rows])), auroc=float(np.mean([r['auroc'] for r in rows])),
                 gamma_state=float(np.mean([r['gamma_state'] for r in rows])))
        for nm in ['stdBKT', 'cBKT', 'g_only', 'correct_only'] + CELLS: o[nm] = cell(rows, nm)
        for nm in ['FB_paper', 'FB_paper_ext', 'FB_fair']: o[nm] = cell(rows, nm)
        o['FB_fair_vs_cBKT'] = cell(rows, 'FB_fair', base='cBKT_ext')
        o['fb_iters'] = [r['FB_fair']['iters'] for r in rows]
        out[scen][sd] = o
        if 'diag' in rows[0]:
            out[scen][sd]['diag'] = rows[0]['diag']
            out[scen][sd]['em_conv'] = rows[0]['em_conv']; out[scen][sd]['em80'] = rows[0]['em']
            out[scen][sd]['em_conv_iters'] = rows[0]['em_conv_iters']
    out[scen]['sizes'] = {}
    for n_, rows in d['sizes'].items():
        out[scen]['sizes'][n_] = {nm: cell(rows, nm) for nm in ['cBKT', 'FB_fair', 'FB_paper']}
        out[scen]['sizes'][n_]['FB_fair_vs_cBKT'] = cell(rows, 'FB_fair', base='cBKT_ext')

# resolution (gamma, AUROC) of the ratings in the rho sweep of cbkt_robust.py, test learners, mean of the 20 groups
import cbkt_robust as rb
from cbkt_revision import gamma, _auc
out['rho_gamma'] = {}
for sig in rb.SIGMAS:
    for rho in rb.RESOLUTIONS:
        g, a = [], []
        for seed in range(rb.SEED0, rb.SEED0 + 20):
            C, Rr, TL, b, tr, te = rb.generate_cont(seed, sig, rho)
            g.append(gamma(Rr[te], C[te])); a.append(_auc(Rr[te], C[te]))
        out['rho_gamma'][f'{sig:.2f}|{rho:.2f}'] = dict(gamma=float(np.mean(g)), auroc=float(np.mean(a)))

json.dump(out, open('results/revision_summary.json', 'w'), indent=1)

f = lambda x: '  nan' if x is None or (isinstance(x, float) and np.isnan(x)) else '%6.1f' % x
for scen, d in out.items():
    if scen == 'rho_gamma': continue
    print(f'\n===== {scen}')
    print(' sd    gamma  auroc  g_state | std   cBKT  chg%  wins | g_only chg% | corr_only chg% | FBpaper chg% (reach) | FBfair chg% (reach) thr | FBfair vs cBKT')
    for sd, o in d.items():
        if sd == 'sizes': continue
        print(f" {sd}  {o['gamma']:.2f}  {o['auroc']:.2f}  {o['gamma_state']:.2f}   | {o['stdBKT']['over']:.2f} {o['cBKT']['over']:.2f} {f(o['cBKT']['pct'])} {o['cBKT']['wins']:2d} |"
              f" {f(o['g_only']['pct'])} | {f(o['correct_only']['pct'])} | {f(o['FB_paper']['pct'])} ({o['FB_paper']['reach']}) ext {f(o['FB_paper_ext']['pct'])} ({o['FB_paper_ext']['reach']}) |"
              f" {f(o['FB_fair']['pct'])} ({o['FB_fair']['reach']}) {o['FB_fair']['thr']:.3f} | {f(o['FB_fair_vs_cBKT']['pct'])} wins {o['FB_fair_vs_cBKT']['wins']} | iters max {max(o['fb_iters'])}")
    o = d['0.08']
    print(' cells (confidence used in this cell only), change vs stdBKT %:')
    for c in CELLS: print('   %-16s %s  wins %d' % (c, f(o[c]['pct']), o[c]['wins']))
    print(' EM 80 iters', np.round(o['em80'], 4), 'converged', np.round(o['em_conv'], 4), 'in', o['em_conv_iters'])
    print(' diag (group 1000, training learners): rating: g_true g_cbkt | s_true s_cbkt | share right/wrong | P(NM|wrong)')
    for k, v in o['diag'].items():
        print('   %s: %.2f %.2f | %.2f %.2f | %.3f %.3f | %.2f' % (k, v['g_true'], v['g_cbkt'], v['s_true'], v['s_cbkt'],
              v['share_correct'], v['share_wrong'], v['p_nm_given_wrong'] or np.nan))
    print(' training sizes: over-practice (reach)   cBKT | FB fair | FB paper | FBfair vs cBKT')
    for n_, v in d['sizes'].items():
        print(f"   n={n_:>4}: {v['cBKT']['over']:.2f} ({v['cBKT']['reach']}) | {v['FB_fair']['over']:.2f} ({v['FB_fair']['reach']}) | "
              f"{v['FB_paper']['over']:.2f} ({v['FB_paper']['reach']}) | {f(v['FB_fair_vs_cBKT']['pct'])} wins {v['FB_fair_vs_cBKT']['wins']}")
