#shared helpers for the 64-channel analysis scripts in ch64/
#
#loads the 64-channel feature tables made by extract_64.py, attaches cohort,
#style, age and recording order info, and holds the small stats helpers that
#every script uses. the classifier pipeline and the bootstrap / sign-flip
#functions are imported from the 14-channel scripts so both analyses match.
#
#note on the log transform: extract_64.py already saves log relative band power
#(the same np.log for every condition), so nothing is logged again here.
#iaf is left in hz, same as the 14-channel analysis.
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


def quiet_blas_warnings():
    #numpy built on apple accelerate gives false "divide by zero / overflow encountered
    #in matmul" warnings (the results are finite). hide only those so the logs stay
    #readable. joblib workers need to call this again inside the worker function.
    warnings.filterwarnings('ignore', message='.*encountered in matmul', category=RuntimeWarning)


quiet_blas_warnings()

#so the existing modules in the repo root can be imported
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import RAW_PATHS
from combined_analysis import clf, cell_weights
from combined_supplementary import bootstrap_ci, signflip

OUT = ROOT / 'results' / 'ch64'
COHORT_FILE = ROOT / 'results' / 'harmonized' / 'braboszcz_classifier_cohort.csv'

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']
FRONTAL = ['F3', 'F4', 'Fz', 'F7', 'F8']
MIDLINE = ['Fz', 'FCz', 'Cz']

#the mind wandering label in each dataset
MW_LABEL = {'braboszcz': 'IMW2', 'brandmeyer': 'MW'}

#braboszcz participants.tsv first_session values, renamed for the tables
ORDER_NAMES = {'meditation': 'MED-first', 'thinking': 'IMW-first'}

#braboszcz styles in the matched cohort
STYLES = ['HYT', 'VIP', 'ISY']


#command line

def add_subset_args(parser):
    #options for small test runs, every script has the same three
    parser.add_argument('--n-brab', type=int, default=None,
                        help='only use this many matched braboszcz meditators (test runs)')
    parser.add_argument('--n-brand', type=int, default=None,
                        help='only use this many brandmeyer subjects (test runs)')
    parser.add_argument('--n-ctr', type=int, default=None,
                        help='only use this many braboszcz ctr subjects (test runs)')


def is_test_run(args):
    return any(v is not None for v in (args.n_brab, args.n_brand, args.n_ctr))


def output_dir(args):
    #test runs write to results/ch64/smoke so they never overwrite real results
    out = OUT / 'smoke' if is_test_run(args) else OUT
    out.mkdir(parents=True, exist_ok=True)
    return out


#loading

def feature_columns(df):
    #feature columns look like "Fz_theta" or "Fz_IAF"
    feats = []
    for c in df.columns:
        parts = c.split('_')
        if len(parts) == 2 and parts[1] in BANDS + ['IAF']:
            feats.append(c)
    return feats


def band_columns(df):
    #only the band power columns (no iaf)
    return [c for c in feature_columns(df) if not c.endswith('_IAF')]


def pick_round_robin(subject_style, n):
    #take subjects one style at a time (hyt, isy, vip, hyt, ...) so even a tiny
    #test run has every style in it
    by_style = {}
    for style, g in subject_style.groupby('style'):
        by_style[style] = sorted(g.subject)
    chosen = []
    while len(chosen) < n and any(by_style.values()):
        for style in sorted(by_style):
            if by_style[style] and len(chosen) < n:
                chosen.append(by_style[style].pop(0))
    return chosen


def load_braboszcz(include_ctr=False, n_brab=None, n_ctr=None):
    b = pd.read_csv(OUT / 'braboszcz_features.csv')

    #matched 54-subject cohort and its style labels
    cohort = pd.read_csv(COHORT_FILE)
    cohort = cohort[cohort.in_classifier_cohort][['subject', 'style']]
    style_of = dict(zip(cohort.subject, cohort['style']))

    #age and recording order come from participants.tsv
    part = pd.read_csv(Path(RAW_PATHS['braboszcz']) / 'participants.tsv', sep='\t')
    part = part.rename(columns={'participant_id': 'subject'})
    b = b.merge(part[['subject', 'age', 'first_session', 'years_of_practice']],
                on='subject', how='left')

    b['matched'] = b.subject.isin(style_of)
    b['style'] = b.subject.map(style_of)
    b.loc[b.group == 'ctr', 'style'] = 'CTR'
    b['order'] = b.first_session.map(ORDER_NAMES)

    #keep the matched meditators, plus ctr if asked
    keep = b.matched.copy()
    if include_ctr:
        keep = keep | (b.group == 'ctr')
    b = b[keep].copy()

    #small subsets for test runs
    if n_brab is not None:
        med = b[b.matched][['subject', 'style']].drop_duplicates()
        chosen = set(pick_round_robin(med, n_brab))
        b = b[~b.matched | b.subject.isin(chosen)]
    if n_ctr is not None:
        ctr = sorted(b.loc[b['style'] == 'CTR', 'subject'].unique())[:n_ctr]
        b = b[(b['style'] != 'CTR') | b.subject.isin(ctr)]
    return b.copy()


def load_brandmeyer(n_brand=None):
    m = pd.read_csv(OUT / 'brandmeyer_features.csv')
    #all brandmeyer participants practice in the himalayan tradition
    m['style'] = 'BRAND'
    m['matched'] = False
    #probe ids restart in every session, so a calibration unit is session + probe
    m['cal_unit'] = m.session.astype(str) + '_' + m.probe_id.astype(str)

    if n_brand is not None:
        #for test runs take subjects that have both med and mw epochs
        counts = m.groupby(['subject', 'condition']).size().unstack()
        both = sorted(counts.dropna().index)[:n_brand]
        m = m[m.subject.isin(both)]
    return m.copy()


def load_features(datasets=('braboszcz', 'brandmeyer'), include_ctr=False,
                  n_brab=None, n_brand=None, n_ctr=None):
    #one table with every epoch, plus the list of the 448 feature columns
    frames = []
    if 'braboszcz' in datasets:
        frames.append(load_braboszcz(include_ctr, n_brab, n_ctr))
    if 'brandmeyer' in datasets:
        frames.append(load_brandmeyer(n_brand))
    d = pd.concat(frames, ignore_index=True, sort=False)

    feats = feature_columns(d)
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan)
    d['subj'] = d.dataset + '_' + d.subject.astype(str)

    #quick check that the band power is already log relative power (always <= 0)
    bands = band_columns(d)
    if np.nanmax(d[bands].to_numpy(float)) > 1e-6:
        raise ValueError('band power is not log relative power, check extract_64.py output')
    return d, feats


def load_from_args(args, datasets=('braboszcz', 'brandmeyer'), include_ctr=False):
    return load_features(datasets, include_ctr, args.n_brab, args.n_brand, args.n_ctr)


def med_vs_mw(d):
    #meditation vs mind wandering: med vs imw2 for braboszcz, med vs mw for brandmeyer
    mw = d.dataset.map(MW_LABEL)
    out = d[(d.condition == 'MED') | (d.condition == mw)].copy()
    out['y'] = (out.condition == 'MED').astype(int)
    return out


def imw2_vs_imw1(d):
    #the label control: second vs first mind wandering run, braboszcz only
    out = d[(d.dataset == 'braboszcz') & d.condition.isin(['IMW1', 'IMW2'])].copy()
    out['y'] = (out.condition == 'IMW2').astype(int)
    return out


#stats

def subject_summary(values, chance=0.5):
    #mean, bootstrap 95% ci over subjects and sign-flip p against chance
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n_subjects=0, mean=np.nan, ci_low=np.nan, ci_high=np.nan, p_vs_chance=np.nan)
    ci = bootstrap_ci(v)
    #signflip() tests against 0.5, so shift the values so that chance sits at 0.5
    p = signflip(v - chance + 0.5)
    return dict(n_subjects=len(v), mean=v.mean(), ci_low=ci[0], ci_high=ci[1], p_vs_chance=p)


def add_fdr(df, pcol, outcol='p_fdr_bh'):
    #benjamini-hochberg over the rows that have a p value
    df = df.copy()
    df[outcol] = np.nan
    ok = df[pcol].notna()
    if ok.sum() > 0:
        df.loc[ok, outcol] = false_discovery_control(df.loc[ok, pcol].to_numpy(), method='bh')
    return df


def standardized_differences(a, b, cols, min_epochs=5):
    #per column (a mean - b mean) / pooled sd, same formula as
    #combined_supplementary.sign_consistency, but for many columns at once.
    #a and b are epoch tables for one subject (condition 1 and condition 0).
    a = a[cols]
    b = b[cols]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-12)
    diff = (a.mean() - b.mean()) / pooled
    #need enough epochs in both conditions, same rule as the 14-channel code
    ok = (a.notna().sum() >= min_epochs) & (b.notna().sum() >= min_epochs) & (pooled > 0)
    diff[~ok] = np.nan
    return diff


def print_table(title, df, n=None):
    print(f'\n{title}')
    if df is None or len(df) == 0:
        print('  (no rows)')
        return
    show = df if n is None else df.head(n)
    print(show.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
