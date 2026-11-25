"""Check the results against the numbers printed in the paper (results/paper_values.json).

  python check_tables.py            # compare results/summary.json (shipped) with the paper
  python check_tables.py --quick    # re-run seed 42 only (about 10 s): Table II rows 1-3 and Table III
  python check_tables.py --rerun    # re-run everything (about 2 min): study -> summary -> compare
  python check_tables.py --rerun-robust   # re-run the additional simulations (about 2 min), then compare everything
  python check_tables.py --rerun-revision # re-run the camera-ready revision analyses (about 9 min), then compare everything

Exit code 0 = every number matches; 1 = at least one differs.
"""
import argparse, json, os, subprocess, sys
import numpy as np

f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'
sgn = lambda v: ('−' if v < 0 else '+') + f'{abs(v):.1f}'

def tables_from_summary(S):
    M, V = S['moderate'], S['severe']
    T2 = [['Operating point', 'stdBKT', 'cBKT moderate', 'cBKT severe'],
          ['Over-practice at 3% under-practice', f2(M['std']['over3']), f2(M['cb']['over3']), f2(V['cb']['over3'])],
          ['Over-practice at 5% under-practice', f2(M['std']['over5']), f2(M['cb']['over5']), f2(V['cb']['over5'])],
          ['Under-practice (%) at 2.5 over-practice', f"{100*M['std']['under25']:.1f}", f"{100*M['cb']['under25']:.1f}", f"{100*V['cb']['under25']:.1f}"]]
    if 'over_mean' in M:
        T2.append(['Over-practice at 5% under-practice, mean of 20 new groups', f2(M['over_mean']['std']), f2(M['over_mean']['cb']), f2(V['over_mean']['cb'])])
    def r3(X, m): d = X[m]; return [f3(d['mae']), f2(d['opps'])] + [f3(x) for x in d['mae_type']]
    T3 = [['Condition', 'Model', 'MAE', 'Opps.', 'MAE calibr.', 'MAE over-conf.', 'MAE under-conf.'],
          ['Moderate', 'stdBKT'] + r3(M, 'std'), ['', 'cBKT'] + r3(M, 'cb'),
          ['Severe', 'stdBKT'] + r3(V, 'std'), ['', 'cBKT'] + r3(V, 'cb')]
    T4 = None
    if 'over_mean' in M:
        def r4(c, mn, X, m, chg=''): return [c, mn, f3(X['thr'][m]), f"{X['over_mean'][m]:.2f} ({X['over_sd'][m]:.2f})", chg]
        T4 = [['Condition', 'Model', 'Threshold', 'Over-practice (SD)', 'Change (%)'],
              r4('Moderate', 'stdBKT', M, 'std'), r4('', 'cBKT', M, 'cb', sgn(-np.mean(M['red']))), r4('', 'Feature-based', M, 'fb', sgn(M['fbchg'])),
              r4('Severe', 'stdBKT', V, 'std'), r4('', 'cBKT', V, 'cb', sgn(-np.mean(V['red']))), r4('', 'Feature-based', V, 'fb', sgn(V['fbchg']))]
    return T2, T3, T4

def sgn1(v): return ('−' if v < 0 else '+') + f'{abs(v):.1f}'

def revision_tables(V, Rb):
    """Table IV (camera-ready: feature-based model refitted) and Table V (robustness checks)."""
    def r4(c, mn, scen, m, chg=True):
        d = V[scen]['0.08'][m]
        return [c, mn, f3(d['thr']), f"{d['over']:.2f} ({d['sd']:.2f})", sgn1(d['pct']) if chg else '']
    T4 = [['Condition', 'Model', 'Threshold', 'Over-practice (SD)', 'Change (%)'],
          r4('Moderate', 'stdBKT', 'moderate', 'stdBKT', False), r4('', 'cBKT', 'moderate', 'cBKT'), r4('', 'Feature-based', 'moderate', 'FB_fair'),
          r4('Severe', 'stdBKT', 'severe', 'stdBKT', False), r4('', 'cBKT', 'severe', 'cBKT'), r4('', 'Feature-based', 'severe', 'FB_fair')]
    g = lambda x: f'{x:.2f}'.replace('-0.00', '0.00').replace('-', '−')
    T5 = [['Check', 'Setting', 'Moderate', 'Severe']]
    for i, sd in enumerate(('0.08', '0.15', '0.25', '0.35')):
        T5.append(['Rating noise SD' if i == 0 else '', sd] +
                  [f"{g(V[s][sd]['gamma'])}, {sgn1(V[s][sd]['cBKT']['pct'])}" for s in ('moderate', 'severe')])
    res, k = Rb['res'], Rb['const']
    for i, rho in enumerate(('1.00', '0.75', '0.50', '0.35', '0.25', '0.00')):
        T5.append(['Rating–answer link ρ' if i == 0 else '', rho] +
                  [f"{g(V['rho_gamma'][f'{sig}|{rho}']['gamma'])}, {sgn1(res[f'{sig}|{rho}']['raw']['pct'])}" for sig in ('0.20', '0.40')])
    T5.append(['Constant in m', '1.25 / 1.50 / 1.75'] +
              [' / '.join(sgn1(k[f'{sig}|1.00|{c}']['pct']) for c in ('1.25', '1.50', '1.75')) for sig in ('0.20', '0.40')])
    T5.append(['Learner bias removed', 'estimate / true bias'] +
              [f"{sgn1(res[f'{sig}|1.00']['A']['pct'])} / {sgn1(res[f'{sig}|1.00']['oracle']['pct'])}" for sig in ('0.20', '0.40')])
    T5.append(['Asymmetric variant', 'guess only / correct answers only'] +
              [f"{sgn1(V[s]['0.08']['g_only']['pct'])} / {sgn1(V[s]['0.08']['correct_only']['pct'])}" for s in ('moderate', 'severe')])
    return T4, T5

def facts_revision(V):
    M, S = V['moderate'], V['severe']; m, s_ = M['0.08'], S['0.08']; c = lambda o, n: sgn1(o[n]['pct'])
    zm, wm = M['sizes'], max(abs(m[n]['pct']) for n in ('cell_low_0', 'cell_mid_0', 'cell_high_0'))
    return {
     "gamma_auroc_main": f"moderate {m['gamma']:.2f} (AUROC {m['auroc']:.2f}); severe {s_['gamma']:.2f} (AUROC {s_['auroc']:.2f})",
     "cells_moderate": f"high-correct {c(m,'cell_high_1')}; low-correct {c(m,'cell_low_1')}; low-wrong {c(m,'cell_low_0')}",
     "cells_severe": f"high-correct {c(s_,'cell_high_1')}; low-correct {c(s_,'cell_low_1')}; low-wrong {c(s_,'cell_low_0')}",
     "wrong_cells_max_abs": f"{max(abs(o[n]['pct']) for o in (m, s_) for n in ('cell_low_0', 'cell_mid_0', 'cell_high_0')):.1f}",
     "fb_wins_vs_cbkt_moderate": f"{m['FB_fair_vs_cBKT']['wins']} of 20",
     "fb_trainsize_moderate": (f"n=200: {zm['200']['FB_fair_vs_cBKT']['pct']:.1f}% vs cBKT ({zm['200']['FB_fair']['reach']} groups reach); "
                               f"n=50: {zm['50']['FB_fair_vs_cBKT']['pct']:+.1f}% ({zm['50']['FB_fair']['reach']}); "
                               f"n=10: {zm['10']['FB_fair']['reach']} groups reach").replace('-', '−'),
     "cbkt_trainsize_moderate": f"{zm['200']['cBKT']['over']:.2f} to {zm['10']['cBKT']['over']:.2f}",
     "fb_noise035_cbkt_wins": f"{20 - M['0.35']['FB_fair_vs_cBKT']['wins']} of 20",
     "stdbkt_em_converged_iter": f"{m['em_conv_iters']}",
     "p_nonmastered_given_wrong_rating1_moderate": f"{m['diag']['1']['p_nm_given_wrong']:.2f}"}

def facts_from_summary(S):
    M, V = S['moderate'], S['severe']
    R = lambda a, b: (1 - a / b) * 100
    ci = lambda X, k: (X['ci'][k][0], X['ci'][k][1], X['ci'][k][2])
    c1, c2 = ci(M, 'over5'), ci(V, 'over5')
    g = lambda X, k: (abs(X['ci'][k][0]), abs(X['ci'][k][2]), abs(X['ci'][k][1]))
    a, b, c, d = g(M, 'mae'), g(M, 'opps'), g(V, 'mae'), g(V, 'opps')
    rm, rv = np.array(M['red']), np.array(V['red']); em = M['em']; sz, zz = M['sizes'], V['sizes']
    return {
     "text_cut5_moderate_pct": f"{R(M['cb']['over5'], M['std']['over5']):.1f}", "text_cut5_severe_pct": f"{R(V['cb']['over5'], V['std']['over5']):.1f}",
     "text_ci_over5_moderate": f"−{abs(c1[0]):.2f} [−{abs(c1[1]):.2f}, −{abs(c1[2]):.2f}]", "text_ci_over5_severe": f"−{abs(c2[0]):.2f} [−{abs(c2[1]):.2f}, −{abs(c2[2]):.2f}]",
     "text_cut3_moderate_pct": f"{R(M['cb']['over3'], M['std']['over3']):.1f}", "text_cut3_severe_pct": f"{R(V['cb']['over3'], V['std']['over3']):.1f}",
     "text_groups_moderate": f"{rm.min():.0f}% to {rm.max():.0f}% (mean {rm.mean():.0f}%)", "text_groups_severe": f"{rv.min():.0f}% to {rv.max():.0f}% (mean {rv.mean():.0f}%)",
     "text_params": f"PG={em[0]:.3f} PS={em[1]:.3f} PT={em[2]:.3f} PL0={em[3]:.3f}",
     "text_mae_gain_moderate": f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]", "text_opps_gain_moderate": f"{b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]",
     "text_mae_gain_severe": f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]", "text_opps_gain_severe": f"{d[0]:.2f} [{d[1]:.2f}, {d[2]:.2f}]",
     "text_severe_smallest_gain_over": f"{f3(V['std']['mae_type'][1])} to {f3(V['cb']['mae_type'][1])}",
     "text_severe_smallest_gain_under": f"{f3(V['std']['mae_type'][2])} to {f3(V['cb']['mae_type'][2])}",
     "text_true_mean_opps": f"{M['true']:.2f}", "text_types_moderate": ", ".join(map(str, M['types'])), "text_types_severe": ", ".join(map(str, V['types'])),
     "text_fb_change_moderate_pct": f"{abs(M['fbchg']):.1f}", "text_cbkt_change_moderate_pct": f"{rm.mean():.1f}", "text_fb_change_severe_pct": f"{V['fbchg']:.0f}",
     "text_fb_groups_reached": f"{M['nfb']} of 20 (moderate); {V['nfb']} of 20 (severe)",
     "text_cbkt_thresholds": f"{M['thr']['cb']:.2f} to {V['thr']['cb']:.2f}",
     "text_trainsize_cbkt_moderate": f"{sz['50']['cBKT'][0]:.2f} to {sz['10']['cBKT'][0]:.2f}", "text_trainsize_cbkt_severe": f"{zz['50']['cBKT'][0]:.2f} to {zz['10']['cBKT'][0]:.2f}",
     "text_trainsize_fb_fail": f"{sz['10']['FB'][1]} of 20 (moderate); {zz['50']['FB'][1]} to {zz['10']['FB'][1]} of 20 (severe)",
     "text_trainsize_cbkt_missed_severe_n10": f"{zz['10']['cBKT'][1]}",
     "abstract_cuts_pct": f"{rm.mean():.0f} and {rv.mean():.0f}"}

def compare(name, got, want, rows):
    bad = 0
    for i, (g, w) in enumerate(zip(got, want)):
        for j, (x, y) in enumerate(zip(g, w)):
            if x != y: bad += 1; rows.append(('FAIL', f'{name} r{i} c{j}', f'paper {y!r}', f'rerun {x!r}'))
    if not bad: rows.append(('ok', name, '', ''))
    return bad

def quick_summary():
    import cbkt_study as st
    S = {}
    for scen in ('moderate', 'severe'):
        r, _ = st.one(42, scen)
        S[scen] = dict(std=r['stdBKT'], cb=r['cBKT'], fb=r['FB'])
    return S

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ap.add_argument('--rerun', action='store_true')
    ap.add_argument('--rerun-robust', action='store_true'); ap.add_argument('--rerun-revision', action='store_true'); a = ap.parse_args()
    P = json.load(open('results/paper_values.json')); rows = []; bad = 0
    if a.rerun:
        subprocess.check_call([sys.executable, 'cbkt_study.py', '--out', 'results/study.json'])
        subprocess.check_call([sys.executable, 'summarize.py'])
    if a.rerun_robust:
        subprocess.check_call([sys.executable, 'cbkt_robust.py'])
    if a.rerun_revision:
        subprocess.check_call([sys.executable, 'cbkt_revision.py'])
        subprocess.check_call([sys.executable, 'summarize_revision.py'], stdout=subprocess.DEVNULL)
    if a.quick:
        S = quick_summary()
        T2, T3, _ = tables_from_summary(S)
        bad += compare('Table II (rows 1-3, seed 42)', T2[:4], P['table_II'][:4], rows)
        bad += compare('Table III (seed 42)', T3, P['table_III'], rows)
    else:
        S = json.load(open('results/summary.json'))
        T2, T3, T4old = tables_from_summary(S)
        bad += compare('Table II', T2, P['table_II'], rows)
        bad += compare('Table III', T3, P['table_III'], rows)
        V = json.load(open('results/revision_summary.json')); Rb = json.load(open('results/robust_summary.json'))
        T4, T5 = revision_tables(V, Rb)
        bad += compare('Table IV (camera-ready)', T4, P['table_IV'], rows)
        bad += compare('Table V (camera-ready)', T5, P['table_V'], rows)
        FV = facts_revision(V); nb = 0
        for k, w in P['facts_revision'].items():
            if FV.get(k) != w: nb += 1; bad += 1; rows.append(('FAIL', k, f'paper {w!r}', f'rerun {FV.get(k)!r}'))
        if not nb: rows.append(('ok', f"{len(P['facts_revision'])} numbers quoted from the revision analyses", '', ''))
        F = facts_from_summary(S)
        PS = P.get('facts_submitted', {}); nb = 0          # submitted version, replaced in the camera-ready version
        if 'table_IV_submitted' in PS:
            nb += compare('Table IV, feature-based rows (submitted version)', [T4old[3], T4old[6]], PS['table_IV_submitted'], rows)
        for k, w in PS.items():
            if k != 'table_IV_submitted' and F.get(k) != w: nb += 1; rows.append(('FAIL', k, f'submitted {w!r}', f'rerun {F.get(k)!r}'))
        if not nb and PS: rows.append(('ok', 'submitted-version numbers of the old feature-based model (superseded)', '', ''))
        bad += nb
        for k, w in P['facts'].items():
            if F.get(k) != w: bad += 1; rows.append(('FAIL', k, f'paper {w!r}', f'rerun {F.get(k)!r}'))
        if all(F.get(k) == w for k, w in P['facts'].items()): rows.append(('ok', f"{len(P['facts'])} numbers quoted in the text", '', ''))
        if 'facts_robust' in P and os.path.exists('results/robust_summary.json'):      # additional simulations (cbkt_robust.py)
            import cbkt_robust
            FR = cbkt_robust.facts_robust(json.load(open('results/robust_summary.json'))); nb = 0
            for k, w in P['facts_robust'].items():
                if FR.get(k) != w: nb += 1; bad += 1; rows.append(('FAIL', k, f'paper {w!r}', f'rerun {FR.get(k)!r}'))
            if not nb: rows.append(('ok', f"{len(P['facts_robust'])} numbers quoted from the additional simulations", '', ''))
    for r in rows: print(*r, sep='   ')
    print('\nALL NUMBERS MATCH THE PAPER' if not bad else f'\n{bad} DIFFERENCE(S) FROM THE PAPER')
    sys.exit(1 if bad else 0)
