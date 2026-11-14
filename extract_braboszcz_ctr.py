"""B1: harmonized extraction (same pipeline as reextract_harmonized.py) for Braboszcz CTR subjects.
Writes results/harmonized/braboszcz_ctr_features.csv (new file; existing feature table untouched).
Resumable: subjects already in the CSV are skipped. Usage: venv/bin/python extract_braboszcz_ctr.py [--limit N]
NOTE: for CTR the 'MED' block (med2) is breath counting, not a tradition-specific practice (dataset README).
"""
import argparse, gc, traceback
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from config import RAW_PATHS
from reextract_harmonized import braboszcz_jobs, prep_paired, fixed_epochs
from braboszcz_feature_extraction import (COMMON_CHANNELS_14, load_raw_bdf, apply_channel_types_from_bids, rename_to_standard_1020)

OUT = Path('results/harmonized/braboszcz_ctr_features.csv'); FAIL = Path('results/harmonized/braboszcz_ctr_extraction_failures.csv')
ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); a = ap.parse_args()
part = pd.read_csv(Path(RAW_PATHS['braboszcz'])/'participants.tsv', sep='\t').set_index('participant_id')
ctr = {s for s in part.index if str(part.loc[s, 'group']).lower() == 'ctr'}
done = set(pd.read_csv(OUT, usecols=['subject']).subject) if OUT.exists() else set()
fails = []; n = 0
with threadpool_limits(limits=1):
    for s, paths, exp in braboszcz_jobs(all_available=True):
        if s not in ctr or s in done: continue
        if a.limit and n >= a.limit: break
        try:
            raws = []
            for cond, bdf, tsv in paths:
                r = load_raw_bdf(str(bdf)); r = apply_channel_types_from_bids(r, str(tsv)); r = rename_to_standard_1020(r)
                r.pick_types(eeg=True, eog=False, misc=False, exclude=[]); r.pick_channels(COMMON_CHANNELS_14, ordered=True); raws.append((cond, r))
            pre, bads, ics = prep_paired(raws)
            fr = pd.concat([fixed_epochs(r, c, s, 'braboszcz', exp, {'bad_channels': ','.join(bads), 'ica_excluded': ','.join(map(str, ics))}) for c, r in pre], ignore_index=True)
            fr['age'] = part.loc[s, 'age']; fr['group'] = part.loc[s, 'group']; fr['years_of_practice'] = part.loc[s, 'years_of_practice']
            fr.to_csv(OUT, mode='a', header=not OUT.exists(), index=False); n += 1
            print('CTR', s, 'ok', fr.groupby('condition').size().to_dict(), flush=True)
        except Exception as e:
            fails.append({'subject': s, 'reason': f'{type(e).__name__}: {e}'}); print('CTR', s, 'FAILED', e, flush=True)
        finally: gc.collect()
if fails: pd.DataFrame(fails).to_csv(FAIL, index=False)
print('DONE', n, 'extracted;', len(fails), 'failed')
