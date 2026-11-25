# cBKT replication code

Simulation code for the paper

> H. Yan, Kinshuk, and F. Lin, "Confidence-Integrated Bayesian Knowledge Tracing for Adaptive Practicing,"
> in *Proc. AIESC 2027*, Tokyo, Japan, 2027.

(confidence-integrated BKT, "cBKT", compared with standard BKT and a feature-based model).
Every number in Tables II–V, in the Results text and in the sentences on the additional simulations comes from this code.
The camera-ready version added the analyses in `cbkt_revision.py` in response to the reviewer (see
[Camera-ready revision analyses](#camera-ready-revision-analyses-cbkt_revisionpy) below).

## Quick start

```bash
pip install -r requirements.txt          # numpy, matplotlib only
python check_tables.py                   # 1 second: shipped results vs. the paper's tables
python check_tables.py --quick           # 10 seconds: re-simulate seed 42, check Table II rows 1-3 and Table III
python check_tables.py --rerun           # about 2 minutes: re-run the whole study, then check everything
python check_tables.py --rerun-robust    # about 2 minutes: re-run the additional simulations, then check everything
python check_tables.py --rerun-revision  # about 9 minutes: re-run the camera-ready revision analyses, then check everything
python make_figures.py                   # redraw Figs. 1-3 into figures/
```

`check_tables.py` prints `ok` for each table and `ALL NUMBERS MATCH THE PAPER` when every value equals the one
printed in the paper (`results/paper_values.json`). It prints `FAIL` lines with both values otherwise
and exits with code 1.

## What each file does

| File | Purpose |
|---|---|
| `cbkt_sim.py` | Core code: data generator (Table I), EM fitting with forward–backward, stdBKT and cBKT tracking, trade-off curves. Running it prints the 5% operating point for seed 42 in both conditions. |
| `cbkt_study.py` | The full study: default sample (seed 42) with 2,000 learner-level bootstrap resamples, 20 new groups (seeds 1000–1019), training-size runs (10, 50, 200 learners), and the feature-based comparison model. Writes `results/study.json`. |
| `cbkt_robust.py` | The additional simulations quoted in the paper: bias as a continuous learner trait, weaker confidence–correctness link (resolution ρ), the constant in m = c − CF, and bias estimated from earlier responses (option A), compared with raw cBKT, own-mean centring (option C) and an oracle. Writes `results/robust.json` and `results/robust_summary.json`. |
| `cbkt_revision.py` | The camera-ready analyses: rating-noise sweep with the resolution of the ratings (gamma, AUROC), where the gain comes from (confidence used for one response type at a time), asymmetric variants, and the feature-based model refitted without the 0.40 cap, to convergence with five restarts. Writes `results/revision.json`. |
| `summarize.py` | Turns `results/study.json` into `results/summary.json` (means, SDs, ranges, thresholds, group counts). |
| `summarize_revision.py` | Turns `results/revision.json` into `results/revision_summary.json`, adds gamma and AUROC for the ρ sweep of `cbkt_robust.py`, and prints all revision results. |
| `check_tables.py` | Compares the results with the paper (see above). |
| `make_figures.py` | Draws Fig. 1 (workflow), Fig. 2 (trade-off curves) and Fig. 3 (three example learners). |
| `results/` | Shipped output (`study.json`, `summary.json`, `robust*.json`, `revision*.json`) and the paper's printed values (`paper_values.json`). |

## Where each paper number comes from

| Paper item | Source in the code |
|---|---|
| Table II, rows 1–3 | `summary.json` → `moderate/severe` → `std`, `cb` (`over3`, `over5`, `under25`), seed 42 |
| Table II, row 4 | the 20 groups (`over_mean`, `over_sd`, `thr`, `red`) |
| Table IV | `revision_summary.json` → `moderate/severe` → `0.08` → `stdBKT`, `cBKT`, `FB_fair` (`over`, `sd`, `thr`, `pct`) |
| Table V | noise rows: `revision_summary.json` (`gamma`, `cBKT.pct`); ρ rows: `revision_summary.json` → `rho_gamma` and `robust_summary.json` → `res` (`raw`); constant: `robust_summary.json` → `const`; bias removed: `res` (`A`, `oracle`); asymmetric: `revision_summary.json` (`g_only`, `correct_only`) |
| Response-type breakdown (Results D) | `revision_summary.json` → `cell_<band>_<answer>` (band low = ratings 1–2, mid = 3, high = 4–5; answer 1 = correct) |
| Table III | seed 42 (`mae`, `opps`, `mae_type`), group sizes (`types`), true mean opportunities (`true`) |
| Bootstrap intervals in the text | `ci` (2,000 resamples, 95% percentile intervals) |
| Parameter recovery | `em` (mean of the 20 groups) |
| Training-size results | `revision_summary.json` → `sizes` (camera-ready); `summary.json` → `sizes` (submitted version) |
| Constant 1.5 (Method) | `robust_summary.json` → `const` (resolution 1.00, `pct` for c = 1.25 … 1.75; c = 1.0 → `reach`) |
| Bias estimated from earlier responses (Discussion A) | `robust_summary.json` → `res` → `0.20\|1.00`, `0.40\|1.00` (`raw`, `A`, `oracle`, `pct`) |
| Weaker confidence–correctness link (Limitations) | `robust_summary.json` → `res` → ρ = 0.50 and ρ = 0.00 (`raw`, `pct`) |

## Simulation in brief

3,000 learners, 15 practice opportunities, a hidden two-state process with P_G = 0.30, P_S = 0.20, P_T = 0.15 and
P_L0 ~ U(0.1, 0.5). Confidence ratings come from correctness and true mastery (Table I of the paper), shifted by a
learner-type bias (calibrated, overconfident, underconfident) and cut into ratings 1–5. Two conditions: moderate
(50/25/25 %, bias 0.25) and severe (30/35/35 %, bias 0.40). Learners are split 70/30 into training and test sets;
parameters are fitted on the training learners with 80 EM iterations and every result is scored on the test learners.
cBKT sets the guess and slip to clip((1.5 − rating/5) · P), with the clip range [0.05, 0.40], and uses the standard
BKT update. The feature-based model learns P(correct | state, rating) as ten free values by EM; it is our
implementation of the FAST-style design described in the paper.

## Additional simulations (`cbkt_robust.py`)

Same hidden process, rating rule and model fitting as the main study, with these differences only:

- **Bias is continuous.** Each learner's bias is b ~ Normal(0, σ_b) with σ_b = 0.2 or 0.4 (close to the moderate and severe conditions), instead of three learner types.
- **Resolution ρ.** The base confidence level θ of Table I (panel C) is moved toward the midpoint, θ′ = 0.5 + ρ(θ − 0.5), before the bias and noise are added. ρ = 1 is Table I; ρ = 0 leaves ratings that carry only bias and noise.
- **Ways of using the rating** (all with the same fitted P_G, P_S, P_T, P_L0): `raw` is cBKT as in the paper; `A` subtracts a bias estimate computed from the learner's *earlier* responses only (the sum of earlier deviations of the rating from the training learners' mean rating for a right or wrong answer, divided by n_earlier + 3; zero at the first opportunity); `C` centres each rating on the learner's own running mean; `oracle` subtracts the true bias.
- **Constant.** The 1.5 in m = 1.5 − CF is replaced by c = 1.0, 1.25, 1.5, 1.75, 2.0 (at ρ = 1 and ρ = 0.5).
- **Sampling.** The data are drawn for all learners at once (learners are exchangeable, so the first 2,100 are the training learners and the last 900 the test learners, with no permutation). Seeds 1000–1019, as for the 20 new groups. The random stream therefore differs from `cbkt_sim.generate`; both follow the same distributions. Changing the stream moves the percentage changes quoted in the paper by a few points, so they should be read as approximate.
- **Scoring.** As in the main study (over-practice at a 5% under-practice tolerance, mean of the per-group percentage changes from stdBKT, groups where a curve never reaches 5% left out), except that ties between curve points are ordered with a stable sort so that the Python package and the web app agree exactly.

## Camera-ready revision analyses (`cbkt_revision.py`)

Same generator, fitting and scoring as the main study (at rating-noise SD 0.08 the generator reproduces
`cbkt_sim.generate` exactly; this is checked when the script starts), 20 new groups (seeds 1000–1019) per setting.

- **Rating resolution.** Rating-noise SD 0.08 (main), 0.15, 0.25 and 0.35. For each setting the Goodman–Kruskal gamma
  and the AUROC between rating and correctness are computed on the test learners. Both need only ratings and answers,
  so they can also be computed from real practice data.
- **Where the gain comes from.** The confidence adjustment is applied to one response type at a time (rating band ×
  answer) and stdBKT is used for all other responses. Two asymmetric variants: guess adjusted, slip fixed at P_S
  (`g_only`); confidence used only after correct answers (`correct_only`).
- **Feature-based model, refitted.** P(correct | state, rating) as ten free values bounded only to [0.001, 0.999];
  EM until the largest parameter change is below 1e-6 (at most 1,000 iterations); best of five starts by training
  log-likelihood; τ grid extended with eleven steps up to 0.9999. Also run with 10, 50 and 200 training learners and
  at every noise level. This replaces the feature-based model of the submitted version (`cbkt_study.fit_fast`, capped
  at 0.40, 80 iterations), which is kept in `cbkt_study.py` so that the submitted numbers can still be checked.
- **EM convergence check.** stdBKT EM run to convergence (tolerance 1e-8) for comparison with the 80-iteration fits.

## Reproducibility notes

- All randomness uses `numpy.random.default_rng(seed)`. Seeds: 42 (default sample), 1000–1019 (20 groups),
  seed + 999 for the bootstrap. The numbers were produced with Python 3.13, NumPy 2.5 and Matplotlib 3.11;
  other versions should give the same numbers, but if `check_tables.py` reports small differences, compare versions first.
- Over-practice is the mean of max(decision − m_i, 0) over truly mastered learners (never declared = full session of 15).
  Under-practice is the share declared before mastery or declared without ever mastering. Operating points are read by
  linear interpolation between the 40 thresholds τ = 0.50 … 0.99. When a model's curve never reaches the 5% target the
  value is missing (NaN) and that group is left out of the mean; the number of such groups is reported in the paper.
- The main study and the additional simulations take about 2 minutes each on a laptop CPU; the revision analyses take about 9 minutes.
- `check_tables.py` also checks the feature-based numbers of the submitted version (marked "superseded"), which the
  camera-ready version replaced.

## Citation

If you use this code, please cite the paper above.

## License

MIT (see `LICENSE`).
