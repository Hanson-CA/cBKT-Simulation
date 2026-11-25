"""Turn results/study.json into results/summary.json (every number used in Tables II-IV and the text).
Usage: python summarize.py [--study results/study.json] [--out results/summary.json]"""
import argparse, json, numpy as np

def summarize(study):
    out = {}
    for scen, d in study.items():
        main, reps = d['main'], d['reps']
        std, cb, fb = main['stdBKT'], main['cBKT'], main['FB']
        ov = {m: np.array([r[m]['over5'] for r in reps], float) for m in ('stdBKT', 'cBKT', 'FB')}
        th = {m: np.array([r[m]['thr5'] for r in reps], float) for m in ('stdBKT', 'cBKT', 'FB')}
        okfb = ~np.isnan(ov['FB'])
        red = 100 * (1 - ov['cBKT'] / ov['stdBKT'])                       # cBKT cut in each group
        fbchg = float(np.mean(100 * (ov['FB'][okfb] / ov['stdBKT'][okfb] - 1)))   # FB change, groups where it reached 5%
        key = {'stdBKT': 'std', 'cBKT': 'cb', 'FB': 'fb'}
        sizes = {}
        for n_, rs in d['sizes'].items():
            sizes[n_] = {}
            for m in ('stdBKT', 'cBKT', 'FB'):
                v = np.array([r[m]['over5'] for r in rs], float)
                sizes[n_][m] = [float(np.nanmean(v)), int(np.isnan(v).sum())]
        out[scen] = dict(
            std=std, cb=cb, fb=fb, ci=d['ci'], types=main['types'], true=main['true_opps'],
            red=red.tolist(), fbchg=fbchg,
            over_mean={key[m]: float(np.nanmean(ov[m])) for m in ov},
            over_sd={key[m]: float(np.nanstd(ov[m], ddof=1)) for m in ov},
            thr={key[m]: float(np.nanmean(th[m])) for m in th},
            nfb=int(okfb.sum()), em=np.mean([r['em'] for r in reps], 0).tolist(),
            cbless=int((ov['cBKT'][okfb] < ov['FB'][okfb]).sum()), sizes=sizes)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='results/study.json'); ap.add_argument('--out', default='results/summary.json')
    a = ap.parse_args()
    json.dump(summarize(json.load(open(a.study))), open(a.out, 'w'))
    print('wrote', a.out)
