"""A4: recording order of Braboszcz runs from BDF header start times (no scans.tsv exists).
Writes results/harmonized/braboszcz_run_order.csv. Header-only reads; no EEG data loaded.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from config import RAW_PATHS
from braboszcz_classifier_cohort import selected_subjects

ROOT = Path(RAW_PATHS['braboszcz']); TASKS = ['think1', 'med1breath', 'med2', 'think2']

def header_start(bdf):
    with open(bdf, 'rb') as f:
        h = f.read(256)
    d, t = h[168:176].decode('ascii').strip(), h[176:184].decode('ascii').strip()
    return datetime.strptime(d + ' ' + t, '%d.%m.%y %H.%M.%S'), (d, t)

rows = []
for sd in sorted(ROOT.glob('sub-*')):
    s = sd.name; row = dict(subject=s, in_matched_cohort=s in selected_subjects())
    start, dur = {}, {}
    for t in TASKS:
        p = sd/'eeg'/f'{s}_task-{t}_eeg.bdf'
        if not p.exists(): continue
        try: start[t], raw = header_start(p)
        except Exception as e: row[f'{t}_header_error'] = str(e)[:60]; continue
        try: dur[t] = float(json.load(open(sd/'eeg'/f'{s}_task-{t}_eeg.json')).get('RecordingDuration'))
        except Exception: dur[t] = None
        row[f'{t}_start'] = start[t].isoformat(); row[f'{t}_dur_s'] = dur[t]
    order = sorted(start, key=start.get); row['order'] = '>'.join(order)
    row['first_block_type'] = ('think' if order and order[0].startswith('think') else 'med') if order else None
    def gap(a, b):  # seconds from END of a to START of b (positive = b after a)
        if a in start and b in start and dur.get(a) is not None:
            return (start[b] - (start[a] + timedelta(seconds=dur[a]))).total_seconds()
    def delta(a, b):  # start-to-start
        if a in start and b in start: return (start[b] - start[a]).total_seconds()
    row['think1_to_think2_start_s'] = delta('think1', 'think2')
    row['think1_end_to_think2_start_s'] = gap('think1', 'think2')
    row['med2_to_think2_start_s'] = delta('med2', 'think2')
    row['med2_end_to_think2_start_s'] = gap('med2', 'think2')
    row['think1_before_think2'] = ('think1' in start and 'think2' in start and start['think1'] < start['think2'])
    row['med2_before_think2'] = ('med2' in start and 'think2' in start and start['med2'] < start['think2'])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv('results/harmonized/braboszcz_run_order.csv', index=False)
m = df[df.in_matched_cohort]
print('subjects', len(df), 'matched', len(m))
print(df.head(4).T.to_string())
print('\norder patterns (matched):'); print(m.order.value_counts().to_string())
print('\nfirst block type (matched):'); print(m.first_block_type.value_counts().to_string())
print('\nthink1 before think2:', m.think1_before_think2.value_counts().to_dict(), '| med2 before think2:', m.med2_before_think2.value_counts().to_dict())
for c in ['think1_to_think2_start_s','think1_end_to_think2_start_s','med2_to_think2_start_s','med2_end_to_think2_start_s']:
    print(c, (m[c]/60).describe()[['count','mean','min','50%','max']].round(1).to_dict(), '(minutes)')
