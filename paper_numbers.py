"""C1: results/paper_numbers.csv - one row per number the paper cites. Every value is read from a result file; nothing is typed in.
Columns: claim_id, analysis, dataset, N, metric, mean, ci_low, ci_high, p, p_fdr, source_file, script. Caveats: results/paper_numbers_notes.csv.
Where a source file has no CI/p for per-subject values, they are computed here with the project's own helpers (bootstrap_ci: 5000 resamples seed 42;
signflip: 20000 sign flips seed 42, test of the mean against 0.5 for accuracies, against 0 for differences).
"""
from pathlib import Path
import numpy as np, pandas as pd
from combined_supplementary import bootstrap_ci, signflip
R = Path('results'); H = R/'harmonized'; rows, notes = [], []
def add(cid, analysis, dataset, N, metric, mean, lo=np.nan, hi=np.nan, p=np.nan, pf=np.nan, src='', script='', note=None):
    rows.append(dict(claim_id=cid, analysis=analysis, dataset=dataset, N=N, metric=metric, mean=mean, ci_low=lo, ci_high=hi, p=p, p_fdr=pf, source_file=src, script=script))
    if note: notes.append(dict(claim_id=cid, note=note))
def acc(v, centre=.5):
    v = np.asarray(v, float); lo, hi = bootstrap_ci(v); return v.mean(), lo, hi, signflip(v if centre == .5 else v+.5)
rd = lambda p: pd.read_csv(p)
# --- LOSO classifiers
for _, r in rd(R/'combined_loso_statistics.csv').query("model=='logistic'").iterrows():
    add(f'LOSO_{r.dataset}_{r.scheme}', 'LOSO logistic feature model', r.dataset, int(r.n_subjects), 'balanced accuracy', r['mean'], r.ci_low, r.ci_high, r.p_vs_chance, r.p_fdr, 'results/combined_loso_statistics.csv', 'combined_analysis.py; combined_supplementary.py')
pr = rd(R/'combined_dataset_probe.csv').iloc[0]; add('dataset_identity_probe', 'dataset identity probe (subject-grouped 5-fold)', 'braboszcz+brandmeyer', int(pr.n_subjects), 'balanced accuracy', pr.value, src='results/combined_dataset_probe.csv', script='combined_analysis.py')
for _, r in rd(R/'combined_loso_paired_statistics.csv').iterrows():
    add(f'LOSO_paired_{r.dataset}_{r.model}_{r.comparison}', 'paired LOSO scheme comparison', r.dataset, int(r.n_subjects), f'mean difference {r.comparison}', r.mean_difference, r.ci_low, r.ci_high, r.p_signflip, r.p_fdr, 'results/combined_loso_paired_statistics.csv', 'combined_supplementary.py') if r.model == 'logistic' else None
# --- blocked within-subject controls (note: control tasks in combined_analysis.main() use untransformed features)
names = {'MED_vs_IMW': 'blocked_MED_vs_IMW2', 'think2_vs_think1': 'blocked_think2_vs_think1', 'think2_first_vs_second_half': 'blocked_think2_halves'}
for _, r in rd(R/'combined_control_statistics.csv').iterrows():
    add(names[r.task], 'blocked within-subject CV', 'braboszcz', int(r.n_subjects), 'balanced accuracy', r['mean'], r.ci_low, r.ci_high, r.p_vs_chance, src='results/combined_control_statistics.csv', script='combined_analysis.py; combined_supplementary.py',
        note='Control tasks in combined_analysis.main() are fit on raw (not log-transformed) features while MED_vs_IMW uses log features; see preproc_sensitivity main row (log-consistent control 0.675 vs 0.671 here).' if r.task != 'MED_vs_IMW' else None)
pc = rd(R/'combined_control_paired_statistics.csv').iloc[0]; add('blocked_real_minus_file_control', 'paired real-vs-file control', 'braboszcz', int(pc.n_subjects), 'mean paired difference in balanced accuracy', pc.mean_paired_difference, pc.ci_low, pc.ci_high, pc.p_signflip, src='results/combined_control_paired_statistics.csv', script='combined_supplementary.py')
# --- calibration
cal = rd(H/'calibration_decomposition_subject.csv')
for task, g in cal.groupby('task'):
    for col in ('base', 'bias_only', 'full_update'):
        m, lo, hi, p = acc(g[col]); add(f'calibration_{task}_{col}', 'calibration (56 s, frac 0.05, 5 reps)', 'braboszcz', len(g), 'balanced accuracy', m, lo, hi, p, src='results/harmonized/calibration_decomposition_subject.csv', script='calibration_decomposition.py',
                                    note='think2_vs_think1 calibration in calibration_decomposition.main() uses untransformed features.' if task != 'MED_vs_IMW2' else None)
    m, lo, hi, p = acc(g.gain_full, 0); add(f'calibration_{task}_gain_full', 'calibration gain (full update - base)', 'braboszcz', len(g), 'balanced accuracy gain', m, lo, hi, p, src='results/harmonized/calibration_decomposition_subject.csv', script='calibration_decomposition.py')
na = rd(H/'calibration_null_adjusted_gain.csv'); m, lo, hi, p = acc(na.null_adjusted_gain, 0); add('calibration_null_adjusted_gain', 'calibration null-adjusted gain', 'braboszcz', len(na), 'gain(MED/IMW2) - gain(think2/think1)', m, lo, hi, p, src='results/harmonized/calibration_null_adjusted_gain.csv', script='calibration_decomposition.py')
if (H/'calibration_curve_summary.csv').exists():
    for _, r in rd(H/'calibration_curve_summary.csv').iterrows():
        add(f'calibration_curve_{r.task.split(" ")[0]}_{int(round(r.calibration_seconds))}s', 'calibration curve (both tasks log-transformed)', 'braboszcz', int(r.N), f'mean gain, ~{r.calibration_seconds:.0f} s calibration data', r.mean_gain_full, r.ci_low, r.ci_high, r.p_signflip, src='results/harmonized/calibration_curve_summary.csv', script='calibration_curve.py')
# --- learning curve, specification curve
for _, r in rd(R/'combined_learning_curve_summary.csv').iterrows():
    add(f'learning_curve_{r.dataset}_{int(r.n_train_subjects)}', 'learning curve', r.dataset, int(r.n_subjects), f'balanced accuracy, {int(r.n_train_subjects)} training subjects', r['mean'], r.ci_low, r.ci_high, src='results/combined_learning_curve_summary.csv', script='combined_supplementary.py')
fs = rd(H/'feature_specification_summary.csv')
for (ds, ft), g in fs.groupby(['dataset', 'features']): add(f'spec_curve_{ds}_{ft}', 'feature specification curve (mean over C)', ds, int(g.n_subjects.iloc[0]), 'balanced accuracy', g['mean'].mean(), g.ci_low.mean(), g.ci_high.mean(), src='results/harmonized/feature_specification_summary.csv', script='feature_specification_curve.py', note='ci_low/ci_high are averages of the per-C interval ends, not a CI of the average.')
# --- sign consistency (66 subjects, pooled braboszcz+brandmeyer)
sc = rd(R/'combined_feature_sign_consistency.csv')
for ch in ('F3', 'F4', 'P4', 'T8', 'F7'):
    r = sc[sc.feature == f'{ch}_theta'].iloc[0]
    add(f'sign_{ch}_theta_fracpos', 'feature sign consistency', 'braboszcz+brandmeyer', int(r.n), 'fraction of subjects with positive standardized MED-minus-nonMED effect', r.frac_positive, p=r.p_binomial, pf=r.p_fdr_bh, src='results/combined_feature_sign_consistency.csv', script='combined_supplementary.py')
    add(f'sign_{ch}_theta_median_d', 'feature sign consistency', 'braboszcz+brandmeyer', int(r.n), 'median standardized difference (d)', r.median_standardized_difference, p=r.p_binomial, pf=r.p_fdr_bh, src='results/combined_feature_sign_consistency.csv', script='combined_supplementary.py')
# --- CSP, Riemann
cs = rd(H/'csp_harmonized_loso_subject.csv')
for c, g in cs.groupby('classifier'): m, lo, hi, p = acc(g.bal_acc); add(f'CSP_{c}_harmonized_loso', 'CSP + classifier LOSO (harmonized epochs)', 'braboszcz', len(g), 'balanced accuracy', m, lo, hi, p, src='results/harmonized/csp_harmonized_loso_subject.csv', script='csp_harmonized_loso.py')
rl = rd(H/'riemann_loso_subject.csv')
for (t, mo), g in rl.groupby(['task', 'model']): m, lo, hi, p = acc(g.bal_acc); add(f'LOSO_{mo}_{t}', 'LOSO, tangent space vs feature model', 'braboszcz', len(g), 'balanced accuracy', m, lo, hi, p, src='results/harmonized/riemann_loso_subject.csv', script='riemann_loso.py')
for _, r in rd(H/'riemann_vs_feature_paired.csv').iterrows(): add(f'riemann_minus_feature_{r.task}', 'paired test Riemannian minus feature model', 'braboszcz', int(r.N), 'mean difference in balanced accuracy', r.mean_riemann_minus_feature, r.ci_low, r.ci_high, r.p_signflip, src='results/harmonized/riemann_vs_feature_paired.csv', script='riemann_loso.py')
w = rl[rl.model == 'feature_logreg'].pivot(index='subject', columns='task', values='bal_acc').dropna(); d = (w.MED_vs_IMW2-w.think2_vs_think1).to_numpy(); lo, hi = bootstrap_ci(d)
add('LOSO_feature_real_minus_file_control', 'paired LOSO real-vs-file control (feature model)', 'braboszcz', len(d), 'mean difference MED_vs_IMW2 - think2_vs_think1', d.mean(), lo, hi, signflip(d+.5), src='results/harmonized/riemann_loso_subject.csv', script='riemann_loso.py')
# --- preprocessing sensitivity
for _, r in rd(H/'preproc_sensitivity.csv').iterrows():
    s, sc_ = 'results/harmonized/preproc_sensitivity.csv', 'preproc_sensitivity.py'
    add(f'preproc_{r.variant}_blocked_real', 'preprocessing sensitivity', 'braboszcz', int(r.N_blocked_real), 'blocked MED/IMW2 balanced accuracy', r.blocked_real, src=s, script=sc_)
    add(f'preproc_{r.variant}_blocked_control', 'preprocessing sensitivity', 'braboszcz', int(r.N_blocked_control), 'blocked think2/think1 balanced accuracy', r.blocked_control, src=s, script=sc_)
    add(f'preproc_{r.variant}_difference', 'preprocessing sensitivity', 'braboszcz', int(r.N_blocked_paired), 'paired real minus control', r.difference, r.difference_ci_low, r.difference_ci_high, r.p_difference_signflip, src=s, script=sc_)
    add(f'preproc_{r.variant}_LOSO', 'preprocessing sensitivity', 'braboszcz', int(r.N), 'LOSO within-dataset balanced accuracy', r.LOSO, r.LOSO_ci_low, r.LOSO_ci_high, r.p_LOSO_vs_chance, src=s, script=sc_)
    add(f'preproc_{r.variant}_theta_fracpos', 'preprocessing sensitivity', 'braboszcz', int(r.N_theta), 'fraction subjects positive, frontal theta MED-IMW2', r.frontal_theta_frac_subjects_positive, src=s, script=sc_)
# --- aperiodic
ap = rd(H/'aperiodic_sign_consistency.csv')
for _, r in ap[ap.feature.str.startswith('frontal_')].iterrows(): add(f'aperiodic_{r.feature}', 'aperiodic control sign consistency', 'braboszcz', int(r.n), f'fraction of subjects positive: {r.measure}', r.frac_positive, p=r.p_binomial, pf=r.p_fdr_bh, src='results/harmonized/aperiodic_sign_consistency.csv', script='aperiodic_theta.py')
# --- B1 trait, B2 dose-response, B6 Brandmeyer
for i, r in rd(H/'trait_ctr_vs_experienced.csv').iterrows():
    if pd.isna(r.p_mwu): continue
    tag = 'matched54' if str(r.comparison).startswith('matched') else 'all56' if str(r.comparison).startswith('all') else 'brandmeyer'
    ch = 'all14' if 'all 14' in str(r.note) else 'frontal'
    add(f'trait_{r.dataset}_{tag}_{ch}_{r.outcome}', 'trait: experienced/expert vs CTR/novice', r.dataset, int(r.n_A+r.n_B), f'Cohen d (A-B), {r.outcome}; A={int(r.n_A)}, B={int(r.n_B)}', r.cohens_d, r.d_ci_low, r.d_ci_high, r.p_mwu, r.p_fdr_bh, 'results/harmonized/trait_ctr_vs_experienced.csv', 'trait_ctr_vs_experienced.py',
        note=f'p = Mann-Whitney; permutation p={r.p_perm:.3g}; age-adjusted group p={r.p_group_age_adjusted:.3g}. CTR "MED" is breath counting.' + (' Secondary analysis (all 14 channels), not the prespecified frontal outcome.' if ch == 'all14' else ''))
for _, r in rd(H/'dose_response_years.csv').iterrows(): add(f'dose_{r.target}', 'dose-response with years of practice', 'braboszcz', int(r.N), f'Spearman rho, {r.target}', r.spearman_rho_years, p=r.p_years, src='results/harmonized/dose_response_years.csv', script='dose_response_years.py', note=f'partial rho given age={r.partial_rho_years_given_age:.3f}, p={r.p_partial:.3g}. Four tests, no multiplicity correction.')
for g_, r in rd(H/'brandmeyer_ratings_by_group.csv').iterrows():
    for q in ('Med', 'MW', 'Sleep'): add(f'brandmeyer_rating_{g_}_{q}', 'Brandmeyer ratings by expertise', 'brandmeyer', int(r.n_subjects), f'mean per-subject rating {q}', r[q], src='results/harmonized/brandmeyer_ratings_by_group.csv', script='brandmeyer_sanity.py', note=f'Table 1 value as recorded in decode_brandmeyer_probes.py: {r[q+"_paper"]} (not independently verified)')
fc = rd(H/'brandmeyer_probe_fold_check.csv'); add('brandmeyer_probes_split_across_folds', 'Brandmeyer probe/fold integrity', 'brandmeyer', int(fc.n_probes.iloc[0]), 'probes split across train/test (max over CVs)', int(fc.probes_split_across_train_test.max()), src='results/harmonized/brandmeyer_probe_fold_check.csv', script='brandmeyer_sanity.py')
# --- run order, cohorts
ro = rd(H/'braboszcz_run_order.csv').query('in_matched_cohort'); 
add('order_think1_end_to_think2_start_s', 'recording order', 'braboszcz', len(ro), 'median gap, end of think1 to start of think2 (s)', ro.think1_end_to_think2_start_s.median(), src='results/harmonized/braboszcz_run_order.csv', script='braboszcz_run_order.py')
add('order_med2_think2_start_to_start_min', 'recording order', 'braboszcz', len(ro), 'median |start-to-start| gap med2 vs think2 (min)', (ro.med2_to_think2_start_s.abs()/60).median(), src='results/harmonized/braboszcz_run_order.csv', script='braboszcz_run_order.py', note='27 subjects think-first, 27 meditation-first; header-derived order agrees with participants.tsv first_session in 92/92 subjects.')
for _, r in rd(H/'cohort_by_analysis.csv').iterrows(): add(f'N_{r.analysis}_{r.dataset}', 'cohort size', r.dataset, int(r.N), 'subjects', int(r.N), src='results/harmonized/cohort_by_analysis.csv', script='cohort_by_analysis.py', note=None if r.exclusion_reasons == 'none' else str(r.exclusion_reasons)[:300])
out = pd.DataFrame(rows); out.to_csv(R/'paper_numbers.csv', index=False); pd.DataFrame(notes).to_csv(R/'paper_numbers_notes.csv', index=False)
dup = out.claim_id[out.claim_id.duplicated()].tolist(); print(len(out), 'rows;', len(notes), 'notes; duplicate ids:', dup)
