"""B2: dose-response of the per-subject MED-minus-IMW2 frontal theta effect (and LOSO accuracy) with years of practice.
Cohort: matched 54-subject Braboszcz classifier cohort (all experienced: HYT/VIP/ISY). Features are the harmonized
log relative-theta values used everywhere else. Frontal theta = mean over F3,F4,F7,F8 of the per-subject standardized
difference (Cohen d, pooled SD) MED - IMW2. Spearman correlations; age covariate via partial Spearman (rank residuals).
Output: results/harmonized/dose_response_years.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import combined_analysis as ca
from config import RAW_PATHS

FRONTAL = ['F3_theta', 'F4_theta', 'F7_theta', 'F8_theta']
d, feats = ca.load_data(); d = d[d.dataset == 'braboszcz']
part = pd.read_csv(Path(RAW_PATHS['braboszcz'])/'participants.tsv', sep='\t').set_index('participant_id')
rows = []
for s, g in d.groupby('subject'):
    a, b = g[g.y == 1], g[g.y == 0]
    if len(a) < 5 or len(b) < 5: continue
    ds = [(a[f].mean()-b[f].mean())/np.sqrt((a[f].var(ddof=1)+b[f].var(ddof=1))/2) for f in FRONTAL]
    rows.append(dict(subject=s, frontal_theta_d=float(np.mean(ds)), years=part.loc[s, 'years_of_practice'], age=part.loc[s, 'age'], style=part.loc[s, 'group']))
S = pd.DataFrame(rows)
loso = pd.read_csv('results/combined_loso_subject_results.csv'); loso = loso[(loso.dataset == 'braboszcz') & (loso.model == 'logistic')]
for sch in ('within', 'pooled', 'cross'):
    S = S.merge(loso[loso.scheme == sch][['subject', 'bal_acc']].assign(subject=lambda x: x.subject.str.replace('braboszcz_', '')).rename(columns={'bal_acc': f'loso_{sch}'}), on='subject', how='left')
S['subject'] = S.subject.astype(str)

def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    r = np.corrcoef(res(rx), res(ry))[0, 1]; n = len(x); t = r*np.sqrt((n-3)/(1-r**2))
    return r, 2*stats.t.sf(abs(t), n-3)

out = []
for target in ['frontal_theta_d', 'loso_within', 'loso_pooled', 'loso_cross']:
    z = S[['years', 'age', target]].dropna(); rho, p = stats.spearmanr(z.years, z[target])
    pr, pp = partial_spearman(z.years.to_numpy(), z[target].to_numpy(), z.age.to_numpy())
    ra, pa = stats.spearmanr(z.age, z[target])
    out.append(dict(target=target, N=len(z), years_min=z.years.min(), years_max=z.years.max(), spearman_rho_years=rho, p_years=p,
                    partial_rho_years_given_age=pr, p_partial=pp, spearman_rho_age=ra, p_age=pa))
O = pd.DataFrame(out); O.to_csv('results/harmonized/dose_response_years.csv', index=False)
S.to_csv('results/harmonized/dose_response_subject_values.csv', index=False)
print(O.round(4).to_string(index=False)); print('years of practice: median', S.years.median(), 'IQR', S.years.quantile([.25, .75]).tolist(), '| age corr w/ years rho=%.2f' % stats.spearmanr(S.years, S.age)[0])
print('frontal theta d: mean %.3f, frac>0 %.2f' % (S.frontal_theta_d.mean(), (S.frontal_theta_d > 0).mean()))
