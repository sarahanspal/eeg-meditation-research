import os
import mne
import numpy as np
import pandas as pd
from preprocessing import preprocess_raw, COMMON_CHS, SAMPLES

#LOADER FOR THE BRANDMEYER OPENNEURO DATASET

def load_openneuro_subject_session(openneuro_path, subj_str, ses_num):
    """
    Returns (epochs, labels) for one subject-session, or (None, None) if skipped.
    epochs: (n_epochs, 14, 1024), labels: (n_epochs,) 1=focused, 0=mind-wandering
    """
    ses_str = f'ses-{ses_num:02d}'
    bdf_path = os.path.join(openneuro_path, subj_str, ses_str, 'eeg',
                             f'{subj_str}_{ses_str}_task-meditation_eeg.bdf')
    evt_path = os.path.join(openneuro_path, subj_str, ses_str, 'eeg',
                             f'{subj_str}_{ses_str}_task-meditation_events.tsv')

    if not os.path.exists(bdf_path) or not os.path.exists(evt_path):
        return None, None

    try:
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  SKIPPED {subj_str} {ses_str} (corrupted file): {e}")
        return None, None

    non_eeg = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
               'GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp','Status']
    raw.drop_channels([ch for ch in non_eeg if ch in raw.ch_names])

    montage = mne.channels.make_standard_montage('biosemi64')
    mapping = dict(zip(raw.ch_names, montage.ch_names))
    raw.rename_channels(mapping)
    raw.set_montage(montage, verbose=False)

    try:
        raw = preprocess_raw(raw)
    except Exception as e:
        print(f"  SKIPPED {subj_str} {ses_str} (preprocessing failed): {e}")
        return None, None

    events_df = pd.read_csv(evt_path, sep='\t')
    stimuli = events_df[events_df['value'] == 128]['onset'].values
    responses = events_df[events_df['trial_type'] == 'response'][['onset', 'value']]

    sfreq = raw.info['sfreq']
    epochs, labels = [], []

    for stim_time in stimuli:
        after = responses[responses['onset'] > stim_time]
        if len(after) == 0:
            continue
        first_resp = after.iloc[0]['value']
        label = 1 if first_resp == 2 else 0

        end_s = int(stim_time * sfreq)
        start_s = max(0, end_s - int(60 * sfreq))
        window = raw.get_data(start=start_s, stop=end_s)

        n_ep = window.shape[1] // SAMPLES
        for i in range(n_ep):
            epoch = window[:, i*SAMPLES:(i+1)*SAMPLES]
            if epoch.shape == (len(COMMON_CHS), SAMPLES):
                epochs.append(epoch)
                labels.append(label)

    if len(epochs) == 0:
        return None, None
    return np.array(epochs), np.array(labels)

#LOADER FOR THE SKY+CONTROL DATASET

import scipy.io

def load_sky_subject(sky_path, subj_idx, timings):
    """
    subj_idx: 0-indexed subject (0-39)
    timings: the loaded 'timings' array from subjects_epochs.mat, shape (40, 4, 40)
    Returns (epochs, labels) or (None, None) if skipped.
    epochs: (n_epochs, 14, 1024), labels: 0=pre-SKY baseline, 1=post-SKY rest
    """
    edf_file = os.path.join(sky_path, f'{subj_idx+1:02d}.edf')
    if not os.path.exists(edf_file):
        return None, None

    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    except Exception as e:
        print(f"  SKIPPED subject {subj_idx+1}: {e}")
        return None, None

    raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})
    legacy = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    raw.rename_channels({k: v for k, v in legacy.items() if k in raw.ch_names})

    try:
        raw = preprocess_raw(raw)
    except Exception as e:
        print(f"  SKIPPED subject {subj_idx+1} (preprocessing failed): {e}")
        return None, None

    epochs, labels = [], []
    for event_idx, label in [(0, 0), (3, 1)]:
        for epoch_idx in range(timings.shape[2]):
            start_s = int(timings[subj_idx, event_idx, epoch_idx])
            end_s = start_s + SAMPLES
            if end_s > raw.n_times:
                continue
            epoch = raw.get_data(start=start_s, stop=end_s)
            if epoch.shape == (len(COMMON_CHS), SAMPLES):
                epochs.append(epoch)
                labels.append(label)

    if len(epochs) == 0:
        return None, None
    return np.array(epochs), np.array(labels)

#LOADER FOR BARBOSZCZ DELORME DATASET

DS3969_TASKS = {
    'med1breath': 1,
    'med2':       1,
    'think1':     0,
    'think2':     0,
}

def load_ds3969_subject(ds3969_path, subj_str):
    """
    Returns (epochs, labels) for one subject, or (None, None) if nothing found.
    epochs: (n_epochs, 14, 1024), labels: 1=meditation, 0=instructed mind-wandering
    """
    subj_eeg_dir = os.path.join(ds3969_path, subj_str, 'eeg')
    if not os.path.exists(subj_eeg_dir):
        return None, None

    all_epochs, all_labels = [], []

    for task_name, label in DS3969_TASKS.items():
        bdf_path = os.path.join(subj_eeg_dir, f'{subj_str}_task-{task_name}_eeg.bdf')
        if not os.path.exists(bdf_path):
            continue

        try:
            raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"  {subj_str} {task_name}: SKIPPED — {e}")
            continue

        non_eeg = [ch for ch in raw.ch_names
                   if ch.startswith('EXG') or ch in
                   ['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status']]
        raw.drop_channels([ch for ch in non_eeg if ch in raw.ch_names])

        if len(raw.ch_names) == 64:
            montage = mne.channels.make_standard_montage('biosemi64')
            mapping = dict(zip(raw.ch_names, montage.ch_names))
            raw.rename_channels(mapping)
            raw.set_montage(montage, verbose=False)

        try:
            raw = preprocess_raw(raw)
        except Exception as e:
            print(f"  {subj_str} {task_name}: preprocessing failed — {e}")
            del raw
            continue

        data = raw.get_data()
        n_ep = data.shape[1] // SAMPLES
        for i in range(n_ep):
            epoch = data[:, i*SAMPLES:(i+1)*SAMPLES]
            if epoch.shape == (len(COMMON_CHS), SAMPLES):
                all_epochs.append(epoch)
                all_labels.append(label)

        del raw, data
        import gc; gc.collect()

    if len(all_epochs) == 0:
        return None, None
    return np.array(all_epochs), np.array(all_labels)