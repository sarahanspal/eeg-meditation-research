"""B1: trait comparison, experienced meditators vs novices/controls, on per-subject frontal theta.
Braboszcz: experienced = matched 54 HYT/VIP/ISY subjects (harmonized table) vs CTR (braboszcz_ctr_features.csv, same pipeline).
  Sensitivity: all harmonized experienced-arm subjects (56, includes 2 ISY not in the matched cohort).
Brandmeyer: expert vs novice among the 15 harmonized subjects, run per state only where both groups have >=5 subjects.
Outcome (prespecified, not selected on the data): mean over F3,F4,F7,F8 of log relative theta (same log transform as combined_analysis);
per-subject mean in each state, and the within-subject MED-minus-IMW standardized effect (Cohen d, pooled SD of epochs, mean over the 4 channels).
Subjects need >=5 epochs in a state. Tests: Mann-Whitney U (two-sided), group-label permutation test on the mean difference (20000, seed 42),
Cohen d with bootstrap 95% CI, rank-biserial r, and an age-adjusted OLS group coefficient. BH-FDR over all rows.
CAVEAT: CTR 'MED' is breath counting (dataset README); experienced MED is tradition-specific practice, so MED-state group contrasts mix trait and task.
Output: results/harmonized/trait_ctr_vs_experienced.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from config import RAW_PATHS
from braboszcz_classifier_cohort import selected_subjects

H = Path('results/harmonized'); FR = ['F3', 'F4', 'F7', 'F8']; RNG = np.random.default_rng(42)
def logt(df, chs): return np.log(df[[f'{c}_theta' for c in chs]].clip(lower=1e-12))

def per_subject(df, pos, neg, chs=FR):
    rows = []
    L = logt(df, chs); df = df.assign(_m=L.mean(axis=1))
    for s, g in df.groupby('subject'):
        a, b = g[g.condition == pos], g[g.condition == neg]; r = dict(subject=s, age=g.age.iloc[0] if 'age' in g else np.nan, group=g.group.iloc[0])
        r[f'theta_{pos}'] = a._m.mean() if len(a) >= 5 else np.nan; r[f'theta_{neg}'] = b._m.mean() if len(b) >= 5 else np.nan
        if len(a) >= 5 and len(b) >= 5:
            la, lb = L.loc[a.index], L.loc[b.index]
            r[f'd_{pos}_minus_{neg}'] = float(np.mean([(la[c].mean()-lb[c].mean())/np.sqrt((la[c].var(ddof=1)+lb[c].var(ddof=1))/2) for c in la]))
        rows.append(r)
    return pd.DataFrame(rows)

def cohen_d(x, y):
    nx, ny = len(x), len(y); sp = np.sqrt(((nx-1)*np.var(x, ddof=1)+(ny-1)*np.var(y, ddof=1))/(nx+ny-2)); return (np.mean(x)-np.mean(y))/sp

def compare(dataset, comp, outcome, A, B, note=''):
    """A = experienced/expert, B = novice/control, DataFrames with columns outcome and age."""
    a, b = A[outcome].dropna().to_numpy(), B[outcome].dropna().to_numpy()
    if len(a) < 5 or len(b) < 5: return dict(dataset=dataset, comparison=comp, outcome=outcome, n_A=len(a), n_B=len(b), note=f'not run: a group has <5 subjects. {note}')
    u = stats.mannwhitneyu(a, b, alternative='two-sided'); d = cohen_d(a, b)
    boot = [cohen_d(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(5000)]
    allv = np.concatenate([a, b]); obs = a.mean()-b.mean(); perm = []
    for _ in range(20000):
        p = RNG.permutation(allv); perm.append(p[:len(a)].mean()-p[len(a):].mean())
    p_perm = (np.sum(np.abs(perm) >= abs(obs))+1)/(len(perm)+1)
    Z = pd.concat([A.assign(g=1), B.assign(g=0)]).dropna(subset=[outcome, 'age'])
    X = np.column_stack([np.ones(len(Z)), Z.g, Z.age]); y = Z[outcome].to_numpy(); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y-X@beta; df = len(y)-3; se = np.sqrt(res@res/df*np.linalg.inv(X.T@X)[1, 1]); p_adj = 2*stats.t.sf(abs(beta[1]/se), df)
    return dict(dataset=dataset, comparison=comp, outcome=outcome, n_A=len(a), n_B=len(b), mean_A=a.mean(), mean_B=b.mean(), cohens_d=d,
                d_ci_low=np.percentile(boot, 2.5), d_ci_high=np.percentile(boot, 97.5), rank_biserial=2*u.statistic/(len(a)*len(b))-1,
                U=u.statistic, p_mwu=u.pvalue, p_perm=p_perm, age_mean_A=A.age.mean(), age_mean_B=B.age.mean(),
                beta_group_age_adjusted=beta[1], p_group_age_adjusted=p_adj, n_age_adjusted=len(Z), note=note)

rows = []
bz = pd.read_csv(H/'braboszcz_all_subjects_features.csv'); ctr = pd.read_csv(H/'braboszcz_ctr_features.csv')
part = pd.read_csv(Path(RAW_PATHS['braboszcz'])/'participants.tsv', sep='\t').set_index('participant_id')
for df in (bz, ctr): df['age'] = df.subject.map(part.age); df['group'] = df.subject.map(part.group)
keep = ['subject', 'condition', 'age', 'group'] + [f'{c}_theta' for c in FR] + [f'{c}_theta' for c in ['C3', 'C4', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8']]
bz, ctr = bz[bz.condition.isin(['MED', 'IMW2'])][keep], ctr[ctr.condition.isin(['MED', 'IMW2'])][keep]
cohorts = {'matched 54 experienced (HYT/VIP/ISY) vs CTR': bz[bz.subject.isin(selected_subjects())],
           'all 56 harmonized experienced-arm subjects vs CTR': bz}
ALL14 = [c[:-6] for c in keep if c.endswith('_theta')]
for name, E in cohorts.items():
    for chs, cl in [(FR, 'frontal F3/F4/F7/F8'), (ALL14, 'all 14 channels')]:
        PE, PC = per_subject(E, 'MED', 'IMW2', chs), per_subject(ctr, 'MED', 'IMW2', chs)
        for outcome in ('theta_MED', 'theta_IMW2', 'd_MED_minus_IMW2'):
            rows.append(compare('braboszcz', name, outcome, PE, PC, note=f'channels: {cl}; CTR MED=breath counting'))
        if cl.startswith('frontal') and name.startswith('matched'): PE.assign(arm='experienced').pipe(lambda x: pd.concat([x, PC.assign(arm='CTR')])).to_csv(H/'trait_ctr_vs_experienced_subject_values.csv', index=False)
bm = pd.read_csv(H/'brandmeyer_all_subjects_features.csv'); bm = bm[bm.condition.isin(['MED', 'MW'])].copy(); bm['group'] = bm.group.astype(str)
cnt = bm.drop_duplicates('subject').group.value_counts().to_dict(); print('Brandmeyer subjects by group (15 surviving):', cnt)
PB = per_subject(bm, 'MED', 'MW')
for outcome in ('theta_MED', 'theta_MW', 'd_MED_minus_MW'):
    rows.append(compare('brandmeyer', 'expert vs novice', outcome, PB[PB.group == 'expert'], PB[PB.group == 'novice'], note='frontal F3/F4/F7/F8'))
R = pd.DataFrame(rows); ok = R.p_mwu.notna(); R['p_fdr_bh'] = np.nan; R.loc[ok, 'p_fdr_bh'] = stats.false_discovery_control(R.loc[ok, 'p_mwu'].to_numpy(), method='bh')
R.to_csv(H/'trait_ctr_vs_experienced.csv', index=False)
pd.set_option('display.width', 250); print(R[['dataset', 'comparison', 'outcome', 'n_A', 'n_B', 'mean_A', 'mean_B', 'cohens_d', 'd_ci_low', 'd_ci_high', 'p_mwu', 'p_perm', 'p_group_age_adjusted', 'p_fdr_bh']].round(4).to_string(index=False))
print(R[R.n_B.lt(5)][['outcome', 'n_A', 'n_B', 'note']].to_string(index=False))
