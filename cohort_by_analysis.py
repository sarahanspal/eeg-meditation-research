"""A3: one row per analysis with N, subject-list hash and exclusion reasons.
Subject lists for each analysis are re-derived from the documented inclusion RULE and then
asserted equal to the subject set actually present in the analysis output file, so a mismatch
between rule and result would raise instead of being papered over.
Writes results/harmonized/cohort_by_analysis.csv and cohort_by_analysis_exclusions.csv.
"""
import hashlib
from pathlib import Path
import numpy as np, pandas as pd
import combined_analysis as ca
from combined_supplementary import load_data as _  # noqa: F401  (same loader used by sign_consistency)

R = Path('results'); H = R/'harmonized'
def h(subs): return hashlib.sha1('\n'.join(sorted(subs)).encode()).hexdigest()[:12]
rows, excl = [], []
def add(analysis, dataset, included, loaded, rule, source):
    included, loaded = set(included), set(loaded)
    rows.append(dict(analysis=analysis, dataset=dataset, N=len(included), n_loaded=len(loaded),
                     subject_list_hash=h(included), inclusion_rule=rule, source=source,
                     exclusion_reasons='; '.join(f'{s}: {r}' for s, r in sorted(excl_now.items())) if excl_now else 'none'))
    for s, r in excl_now.items(): excl.append(dict(analysis=analysis, dataset=dataset, subject=s, reason=r))

d, feats = ca.load_data()                      # matched cohort: 54 Braboszcz + 15 Brandmeyer loaded
cnt = d.groupby(['dataset', 'subj', 'condition']).size().unstack(fill_value=0)
b_loaded = set(d[d.dataset == 'braboszcz'].subj); m_loaded = set(d[d.dataset == 'brandmeyer'].subj)

# 1. LOSO classifier (within/pooled/cross): subject needs both classes among its own epochs
loso = pd.read_csv(R/'combined_loso_subject_results.csv')
for ds, loaded in [('braboszcz', b_loaded), ('brandmeyer', m_loaded)]:
    rule = {s: cnt.loc[(ds, s)] for s in loaded}
    inc = {s for s, c in rule.items() if (c.get('MED', 0) > 0) and ((c.get('IMW2', 0) + c.get('MW', 0)) > 0)}
    excl_now = {s: f"single class in held-out data (MED={int(rule[s].get('MED',0))}, other={int(rule[s].get('IMW2',0)+rule[s].get('MW',0))})" for s in loaded - inc}
    seen = set(loso[(loso.dataset == ds) & (loso.model == 'logistic')].subject)
    assert inc == seen, (ds, inc ^ seen)
    add('combined_loso', ds, inc, loaded, 'both MED and non-MED epochs present', 'combined_loso_subject_results.csv')

# 2. Blocked controls (k=5 folds, min_n=20 epochs per class)
bc = pd.read_csv(R/'combined_blocked_controls.csv')
hf = pd.read_csv(H/'braboszcz_all_subjects_features.csv', usecols=['subject', 'condition', 'epoch_idx'])
sel = {s.replace('braboszcz_', '') for s in b_loaded}; hf = hf[hf.subject.isin(sel)]
c2 = hf.groupby(['subject', 'condition']).size().unstack(fill_value=0)
half = hf[hf.condition == 'IMW2'].copy(); half['H'] = np.where(half.epoch_idx < half.groupby('subject').epoch_idx.transform('median'), 'H1', 'H2')
ch = half.groupby(['subject', 'H']).size().unstack(fill_value=0)
specs = {'MED_vs_IMW': lambda s: (c2.loc[s, 'MED'], c2.loc[s, 'IMW2'], 'MED', 'IMW2'),
         'think2_vs_think1': lambda s: (c2.loc[s, 'IMW2'], c2.loc[s, 'IMW1'], 'IMW2', 'IMW1'),
         'think2_first_vs_second_half': lambda s: (ch.loc[s, 'H1'], ch.loc[s, 'H2'], 'H1', 'H2')}
sets = {}
for task, fn in specs.items():
    inc, excl_now = set(), {}
    for s in sorted(sel):
        n1, n2, a, b = fn(s)
        if min(n1, n2) >= 20: inc.add('braboszcz_' + s)
        else: excl_now['braboszcz_' + s] = f'<20 epochs in a class ({a}={int(n1)}, {b}={int(n2)}); blocked 5-fold CV needs >=20/class'
    seen = set(bc[bc.task == task].subject); assert inc == seen, (task, inc ^ seen)
    sets[task] = inc
    add(f'blocked_control:{task}', 'braboszcz', inc, b_loaded, 'min class epochs >= 20', 'combined_blocked_controls.csv')

# 3. Paired real-vs-file control = intersection of the two tasks above
inter = sets['MED_vs_IMW'] & sets['think2_vs_think1']
excl_now = {s: 'missing from MED_vs_IMW (see above)' for s in b_loaded - inter}
pc = pd.read_csv(R/'combined_control_paired_statistics.csv'); assert len(inter) == int(pc.n_subjects[0])
add('paired_control_MED_vs_IMW_minus_think2_vs_think1', 'braboszcz', inter, b_loaded, 'subject in both blocked analyses', 'combined_control_paired_statistics.csv')

# 4. Sign consistency: >=5 epochs in each class, finite non-zero pooled SD, per feature; pooled across datasets
sc = pd.read_csv(R/'combined_feature_sign_consistency.csv')
inc_ds = {}
for ds, loaded in [('braboszcz', b_loaded), ('brandmeyer', m_loaded)]:
    excl_now, inc = {}, set()
    for s in sorted(loaded):
        g = d[d.subj == s]; n1, n0 = int((g.y == 1).sum()), int((g.y == 0).sum())
        if n1 >= 5 and n0 >= 5: inc.add(s)
        else: excl_now[s] = f'<5 epochs in a class (MED={n1}, non-MED={n0})'
    inc_ds[ds] = inc
    add('feature_sign_consistency', ds, inc, loaded, '>=5 epochs per class', 'combined_feature_sign_consistency.csv (n column)')
tot = sum(len(v) for v in inc_ds.values())
assert tot == int(sc.n.max()) == int(sc.n.min()), (tot, sc.n.min(), sc.n.max())

# 5. Other subject-level analyses present in results/harmonized
for name, f in [('calibration_decomposition', 'calibration_decomposition_subject.csv'), ('csp_harmonized_loso', 'csp_harmonized_loso_subject.csv')]:
    p = H/f
    if p.exists():
        x = pd.read_csv(p); excl_now = {}
        subs = {s if s.startswith('braboszcz_') else f'braboszcz_{s}' for s in x.subject.astype(str)}
        add(name, 'braboszcz', subs & b_loaded, b_loaded, 'matched 54-subject Braboszcz cohort', f)

out = pd.DataFrame(rows); out.to_csv(H/'cohort_by_analysis.csv', index=False)
pd.DataFrame(excl).to_csv(H/'cohort_by_analysis_exclusions.csv', index=False)
print(out[['analysis', 'dataset', 'N', 'n_loaded', 'subject_list_hash', 'exclusion_reasons']].to_string(index=False, max_colwidth=90))
