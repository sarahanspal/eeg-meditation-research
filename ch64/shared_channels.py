"""step 1: print the eeg channels braboszcz and brandmeyer share
after the existing 10-20 renaming. nothing is saved, just printed."""
import sys
import time
from pathlib import Path
import mne

# run from the repo root so the existing modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from braboszcz_feature_extraction import (
    load_raw_bdf, apply_channel_types_from_bids, rename_to_standard_1020,
)
from decode_brandmeyer_probes import resolve_dir, brandmeyer_sessions
from config import RAW_PATHS

# same non-eeg channel names that reextract_harmonized.py drops for brandmeyer
BRAND_NON_EEG = ('Status', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp')


def retry(fn, tries=6):
    # google drive reads fail at random, so just wait a bit and try again
    for i in range(tries):
        try:
            return fn()
        except (OSError, TimeoutError) as e:
            print(f'drive read failed ({e}), retry {i + 1}/{tries}')
            time.sleep(10)
    raise RuntimeError('gave up after repeated drive errors')


def braboszcz_channels():
    # google drive sometimes fails to stream a file, so try subjects until one loads
    root = Path(RAW_PATHS['braboszcz'])
    for sub in sorted(p.name for p in root.glob('sub-*') if p.is_dir()):
        eeg = root / sub / 'eeg'
        try:
            raw = load_raw_bdf(str(eeg / f'{sub}_task-med2_eeg.bdf'))
            raw = apply_channel_types_from_bids(raw, str(eeg / f'{sub}_task-med2_channels.tsv'))
        except (OSError, TimeoutError) as e:
            print(f'skipping {sub}: {e}')
            continue
        raw = rename_to_standard_1020(raw)
        raw.pick_types(eeg=True, eog=False, misc=False, exclude=[])
        return sub, raw.ch_names
    raise RuntimeError('no braboszcz subject could be read')


def brandmeyer_channels():
    # use the first session that opens (same drive retry idea as above)
    root = resolve_dir(Path(RAW_PATHS['brandmeyer']).parent, Path(RAW_PATHS['brandmeyer']).name.strip())
    # subject list comes from the folder names (participants.tsv is not needed here)
    subjects = sorted(p.name for p in root.glob('sub-*') if p.is_dir())
    rows = [r for r in brandmeyer_sessions(root, subjects) if r['eeg'] is not None]
    readers = {'.bdf': mne.io.read_raw_bdf, '.edf': mne.io.read_raw_edf,
               '.set': mne.io.read_raw_eeglab, '.vhdr': mne.io.read_raw_brainvision}
    for row in rows:
        ext = Path(row['eeg']).suffix.lower()
        try:
            raw = readers[ext](str(row['eeg']), preload=False, verbose=False)
        except (OSError, TimeoutError) as e:
            print(f"skipping {row['subject']}: {e}")
            continue
        # drop aux channels, then call everything left eeg
        drop = [c for c in raw.ch_names if c.upper().startswith('EXG') or c in BRAND_NON_EEG]
        raw.drop_channels(drop)
        raw.set_channel_types({c: 'eeg' for c in raw.ch_names}, verbose=False)
        raw = rename_to_standard_1020(raw)
        return row['subject'], raw.ch_names
    raise RuntimeError('no brandmeyer session could be read')


if __name__ == '__main__':
    bsub, bchs = retry(braboszcz_channels)
    msub, mchs = retry(brandmeyer_channels)
    print(f'braboszcz ({bsub}): {len(bchs)} eeg channels')
    print(f'brandmeyer ({msub}): {len(mchs)} eeg channels')

    # keep braboszcz order so the list is stable
    shared = [c for c in bchs if c in set(mchs)]
    print(f'\nshared channels: {len(shared)}')
    print(shared)
    print('\nonly in braboszcz:', [c for c in bchs if c not in set(mchs)])
    print('only in brandmeyer:', [c for c in mchs if c not in set(bchs)])

    # make sure the midline channels we care about are there
    for ch in ['Fz', 'FCz', 'Cz', 'Pz']:
        print(f'{ch} in shared list:', ch in shared)
