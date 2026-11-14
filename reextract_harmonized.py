"""Re-extract Braboszcz, Brandmeyer, and SKY with shared EEG preprocessing.
Outputs are new files under results/harmonized; source tables are untouched.
Defaults to all available subjects. Use --limit 2 for a small pipeline smoke run.
"""
from __future__ import annotations
import argparse, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd, mne
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from braboszcz_feature_extraction import (
    COMMON_CHANNELS_14, CANONICAL_BANDS, TARGET_SFREQ, EPOCH_LEN_SEC,
    load_raw_bdf, apply_channel_types_from_bids, rename_to_standard_1020,
    detect_bad_segments, detect_bad_channels_by_variance, clean_bad_channels,
    highpass_filter, compute_psd, band_power, individual_alpha_frequency,
    asymmetry_index, cross_band_ratios,
)
from decode_brandmeyer_probes import decode_probes, resolve_dir, brandmeyer_sessions
from config import RAW_PATHS

OUT=Path('results/harmonized');OUT.mkdir(parents=True,exist_ok=True)
LINE_NOTCH=(50.0,60.0)
HP=1.0


def standardize_and_clean(raw, label, annotate=True):
    """Apply common notch, bad channel, reference, filter, and resample."""
    if annotate:
        bad_ann=detect_bad_segments(raw)
        # The project's fixed dB thresholds mark essentially all SKY EDF
        # samples as bad because its native amplitudes differ from BDF units.
        # Keep those epochs available and record this dataset-specific detector
        # limitation; downstream epoch boundaries/ICA remain shared.
        if len(bad_ann) and bad_ann.duration.sum() < 0.95*(raw.n_times/raw.info['sfreq']):
            raw.set_annotations(bad_ann)
        elif len(bad_ann):
            warnings.warn(f'{label}: artifact detector flagged >=95% of recording; annotations omitted')
    raw.notch_filter(LINE_NOTCH, picks='eeg', verbose=False)
    bad=detect_bad_channels_by_variance(raw,z_thresh=6.0)
    return raw,bad

def prep_paired(raws, annotate=True):
    """Equalize runs and apply the union of bad channels and shared reference."""
    if not raws: return raws, [], []
    bad_union=set()
    for _,r in raws:
        _,b=standardize_and_clean(r,'',annotate=annotate)
        bad_union.update(b)
    processed=[]
    for name,r in raws:
        common=set(r.ch_names)
        # Only channels present in all runs are kept, so reference and ICA use a stable montage.
        common &= set.intersection(*(set(x.ch_names) for _,x in raws))
        r.pick_channels([c for c in r.ch_names if c in common],ordered=True)
        active_bad=sorted(bad_union & set(r.ch_names))
        r.info['bads']=active_bad
        if active_bad:
            pos=(r.get_montage().get_positions().get('ch_pos',{}) if r.get_montage() else {})
            located=[ch for ch in active_bad if ch in pos and np.isfinite(pos[ch]).all()]
            unlocated=[ch for ch in active_bad if ch not in located]
            # Keep the common schema. Sensors without montage positions (for
            # example auxiliary SKY channels) stay marked bad and are excluded
            # from the average reference and ICA; every common feature channel
            # has a standard position and can be interpolated.
            r.info['bads']=located
            if located:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    r.interpolate_bads(reset_bads=True,verbose=False)
                for ch in located:
                    x=r.get_data(picks=[ch])
                    if not np.isfinite(x).all() or np.std(x)<1e-12:
                        raise RuntimeError(f'bad-channel interpolation invalid for {ch}')
            r.info['bads']=unlocated
        missing=[ch for ch in COMMON_CHANNELS_14 if ch not in r.ch_names]
        if missing:raise RuntimeError(f'missing common EEG channels before reference: {missing}')
        r.pick_channels(COMMON_CHANNELS_14,ordered=True)
        r.set_eeg_reference('average',projection=False,verbose=False)
        r.filter(l_freq=HP,h_freq=None,method='iir',iir_params=dict(order=6,ftype='butter'),phase='zero',picks='eeg',verbose=False)
        if not np.isclose(r.info['sfreq'],TARGET_SFREQ):r.resample(TARGET_SFREQ,npad='auto',verbose=False)
        processed.append((name,r))
    # shared ICA fit across paired runs, then applied to each run
    cat=mne.concatenate_raws([r.copy() for _,r in processed],preload=True,verbose=False)
    cat_f=cat.copy().filter(l_freq=1.,h_freq=None,verbose=False)
    ica=mne.preprocessing.ICA(n_components=.99,method='infomax',fit_params=dict(extended=True),random_state=97,max_iter='auto')
    ica.fit(cat_f,decim=2,verbose=False)
    if not np.isfinite(ica.unmixing_matrix_).all() or not np.isfinite(ica.mixing_matrix_).all():
        raise RuntimeError('ICA produced non-finite mixing matrices')
    exclude=set()
    for fn in (lambda:ica.find_bads_eog(cat,ch_name=[c for c in ('Fp1','Fp2') if c in cat.ch_names],verbose=False)[0],
               lambda:ica.find_bads_muscle(cat,verbose=False)[0]):
        try: exclude.update(fn())
        except Exception as e: warnings.warn(f'ICA auto-labeling failed: {e}')
    ica.exclude=sorted(exclude)
    for _,r in processed: ica.apply(r,verbose=False)
    return processed,sorted(bad_union),ica.exclude

def feature_rows(epochs, dataset, subject, condition, experience=None, extra=None):
    freqs,psd=compute_psd(epochs);chs=epochs.ch_names
    b=pd.DataFrame(index=np.arange(len(epochs)))
    for band,rng in CANONICAL_BANDS.items():
        bp=band_power(freqs,psd,rng)
        for ci,ch in enumerate(chs):b[f'{ch}_{band}']=bp[:,ci]
    iaf=individual_alpha_frequency(freqs,psd)
    for ci,ch in enumerate(chs):b[f'{ch}_IAF']=iaf[:,ci]
    f=pd.concat([b,asymmetry_index(b),cross_band_ratios(b,chs)],axis=1)
    f.insert(0,'epoch_idx',np.arange(len(f)));f.insert(0,'condition',condition)
    f.insert(0,'subject',subject);f.insert(0,'dataset',dataset)
    if experience is not None:f['experience_group']=experience
    if extra:
        for k,v in extra.items():f[k]=v
    return f

def fixed_epochs(raw, condition, subject, dataset, experience=None, extra=None):
    ev=mne.make_fixed_length_events(raw,duration=EPOCH_LEN_SEC)
    meta=pd.DataFrame({'condition':condition,'epoch_idx':np.arange(len(ev))})
    ep=mne.Epochs(raw,ev,tmin=0,tmax=EPOCH_LEN_SEC-1/raw.info['sfreq'],baseline=None,preload=True,reject_by_annotation=True,metadata=meta,verbose=False)
    ep.pick_channels([c for c in COMMON_CHANNELS_14 if c in ep.ch_names],ordered=True)
    return feature_rows(ep,dataset,subject,condition,experience,extra)

def braboszcz_jobs(all_available=False):
    root=Path(RAW_PATHS['braboszcz']);p=pd.read_csv(root/'participants.tsv',sep='\t')
    cohort=None if all_available else set(pd.read_csv('braboszcz_all_subjects_features.csv',usecols=['subject']).subject.astype(str).unique())
    years=dict(zip(p.participant_id,p['years_of_practice'])) if 'years_of_practice' in p else {}
    expmap={}
    for sid,val in years.items():
        try: expmap[sid]='experienced' if float(val)>=3 else 'novice'
        except (TypeError,ValueError): pass
    subs=sorted(x.name for x in root.glob('sub-*') if x.is_dir())
    if cohort is not None:subs=[s for s in subs if s in cohort]
    for s in subs:
        eeg=s and root/s/'eeg'
        paths=[]
        for cond,task in [('MED','med2'),('IMW2','think2'),('IMW1','think1')]:
            bdf=eeg/f'{s}_task-{task}_eeg.bdf';tsv=eeg/f'{s}_task-{task}_channels.tsv'
            if bdf.exists() and tsv.exists():paths.append((cond,bdf,tsv))
        present={c for c,_,_ in paths}
        if {'MED','IMW2'}.issubset(present):yield s,paths,expmap.get(s)

def run_braboszcz(limit=None,all_available=False):
    out=[];fail=[]
    for i,(s,paths,exp) in enumerate(braboszcz_jobs(all_available=all_available)):
        if limit and i>=limit:break
        try:
            raws=[]
            for cond,bdf,tsv in paths:
                r=load_raw_bdf(str(bdf));r=apply_channel_types_from_bids(r,str(tsv));r=rename_to_standard_1020(r)
                r.pick_types(eeg=True,eog=False,misc=False,exclude=[]);
                missing=[c for c in COMMON_CHANNELS_14 if c not in r.ch_names]
                if missing:raise RuntimeError(f'{s}/{cond} missing feature channels: {missing}')
                r.pick_channels(COMMON_CHANNELS_14,ordered=True);raws.append((cond,r))
            pre,bads,ics=prep_paired(raws)
            for cond,r in pre:
                f=fixed_epochs(r,cond,s,'braboszcz',exp,{'bad_channels':','.join(bads),'ica_excluded':','.join(map(str,ics))})
                out.append(f)
            print('Braboszcz',s,'ok',flush=True)
        except Exception as e:fail.append({'subject':s,'reason':f'{type(e).__name__}: {e}'});print('Braboszcz',s,'FAILED',e,flush=True)
        finally:gc.collect()
    return out,fail

def run_brandmeyer(limit=None,all_available=False):
    root=resolve_dir(Path(RAW_PATHS['brandmeyer']).parent,Path(RAW_PATHS['brandmeyer']).name.strip())
    pt=pd.read_csv(root/'participants.tsv',sep='\t');exp=dict(zip(pt.participant_id,pt.group));ages=dict(zip(pt.participant_id,pt.age))
    subjects=sorted(pt.participant_id.astype(str).unique())
    if not all_available:
        cohort=set(pd.read_csv('brandmeyer_all_subjects_features.csv',usecols=['subject']).subject.astype(str).unique())
        subjects=[s for s in subjects if s in cohort]
    if limit:subjects=subjects[:limit]
    jobs=brandmeyer_sessions(root,subjects);by={}
    for row in jobs:
        if row['eeg'] is not None and row['events'] is not None:by.setdefault(row['subject'],[]).append(row)
    out=[];fail=[]
    for i,(s,sessions) in enumerate(sorted(by.items())):
        if limit and i>=limit:break
        try:
            raws=[];decoded={}
            for j,row in enumerate(sessions):
                ext=Path(row['eeg']).suffix.lower()
                if ext=='.bdf':r=mne.io.read_raw_bdf(str(row['eeg']),preload=True,verbose=False)
                elif ext=='.edf':r=mne.io.read_raw_edf(str(row['eeg']),preload=True,verbose=False)
                elif ext=='.set':r=mne.io.read_raw_eeglab(str(row['eeg']),preload=True,verbose=False)
                elif ext=='.vhdr':r=mne.io.read_raw_brainvision(str(row['eeg']),preload=True,verbose=False)
                else:raise ValueError(f'unsupported Brandmeyer file format: {ext}')
                # Exclude BioSemi auxiliary/non-EEG channels; map the 64 EEG channels to standard positions.
                drop=[c for c in r.ch_names if c.upper().startswith('EXG') or c in ('Status','GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp')]
                if drop:r.drop_channels(drop)
                r.set_channel_types({c:'eeg' for c in r.ch_names},verbose=False)
                r=rename_to_standard_1020(r);
                missing=[c for c in COMMON_CHANNELS_14 if c not in r.ch_names]
                if missing:raise RuntimeError(f'{s} missing feature channels: {missing}')
                r.pick_channels(COMMON_CHANNELS_14,ordered=True);raws.append((str(j),r));decoded[str(j)]=decode_probes(row['events'])
            pre,bads,ics=prep_paired(raws)
            for key,r in pre:
                probes=decoded[key];probes=probes[probes.label.isin(['MED','MW'])]
                sf=r.info['sfreq'];events=[];metadata=[]
                for _,p in probes.iterrows():
                    for sub_epoch,off in enumerate((-8.05,-4.05)):
                        start=int(round((p.t_probe+off)*sf))
                        if start<0 or start+int(EPOCH_LEN_SEC*sf)>r.n_times:continue
                        events.append([start,0,1]);metadata.append({'condition':p.label,'probe_id':int(p.probe_id),'sub_epoch':sub_epoch,'t_start':start/sf})
                if not events:continue
                ep=mne.Epochs(r,np.asarray(events),event_id=1,tmin=0,tmax=EPOCH_LEN_SEC-1/sf,baseline=None,preload=True,reject_by_annotation=True,metadata=pd.DataFrame(metadata),verbose=False)
                ep.pick_channels([c for c in COMMON_CHANNELS_14 if c in ep.ch_names],ordered=True)
                f=feature_rows(ep,'brandmeyer',s,ep.metadata.condition.to_numpy(),exp.get(s),{'session':key,'age':ages.get(s),'bad_channels':','.join(bads),'ica_excluded':','.join(map(str,ics))})
                # condition differs rowwise; feature_rows sets scalar but dataframe assignment broadcasts arrays fine.
                f['condition']=ep.metadata.condition.to_numpy();f['probe_id']=ep.metadata.probe_id.to_numpy();f['sub_epoch']=ep.metadata.sub_epoch.to_numpy();f['t_start']=ep.metadata.t_start.to_numpy()
                out.append(f)
            print('Brandmeyer',s,'ok',flush=True)
        except Exception as e:fail.append({'subject':s,'reason':f'{type(e).__name__}: {e}'});print('Brandmeyer',s,'FAILED',e,flush=True)
        finally:gc.collect()
    return out,fail

def run_sky(limit=None, only_subjects=None):
    root=Path(RAW_PATHS['sky']);m=sio.loadmat(root/'subjects_epochs.mat');timings=np.asarray(m['timings']);meta=pd.read_csv(root/'SubjectMetaData.csv')
    # Metadata table carries the year-based experience label map used by project analyses.
    expmap={}
    try:
        from experience_mapping import build_sky_experience_map
        expmap=build_sky_experience_map(str(root))
    except Exception as e: print('SKY experience metadata warning:',e)
    out=[];fail=[]
    for si in range(min(timings.shape[0],limit or timings.shape[0])):
        s=f'sky-{si+1:02d}';path=root/f'{si+1:02d}.edf'
        if only_subjects and s not in only_subjects:continue
        if not path.exists():continue
        try:
            r=mne.io.read_raw_edf(str(path),preload=True,verbose=False)
            # Normalize case and legacy 10-20 labels.
            upper={c:c.upper() for c in r.ch_names};r.rename_channels(upper)
            rename={'FP1':'Fp1','FP2':'Fp2','FZ':'Fz','CZ':'Cz','PZ':'Pz','T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
            r.rename_channels({k:v for k,v in rename.items() if k in r.ch_names})
            r.set_channel_types({c:'eeg' for c in r.ch_names},verbose=False)
            r.set_montage(mne.channels.make_standard_montage('standard_1020'),on_missing='ignore',verbose=False)
            missing=[c for c in COMMON_CHANNELS_14 if c not in r.ch_names]
            if missing:raise RuntimeError(f'{s} missing feature channels: {missing}')
            r.pick_channels(COMMON_CHANNELS_14,ordered=True)
            prep,bads,ics=prep_paired([('all',r)],annotate=False)
            r=prep[0][1];sf=r.info['sfreq']
            for row,cond in [(0,'pre'),(3,'post')]:
                starts=timings[si,row].astype(int);events=[];md=[]
                for ix,st in enumerate(starts):
                    # Timings in subjects_epochs.mat are seconds (their range reaches the raw
                    # recording duration), so convert seconds to current samples.
                    st=int(round(st*sf))
                    if st<0 or st+int(EPOCH_LEN_SEC*sf)>r.n_times:continue
                    events.append([st,0,1]);md.append({'condition':cond,'epoch_idx':ix,'t_start':st/sf})
                if not events:continue
                # Some timing rows repeat an epoch start. Keep the first occurrence;
                # MNE requires event samples to be unique and repeated windows add no data.
                seen=set();unique_events=[];unique_md=[]
                for event,meta_row in zip(events,md):
                    if event[0] in seen:continue
                    seen.add(event[0]);unique_events.append(event);unique_md.append(meta_row)
                events,md=unique_events,unique_md
                ep=mne.Epochs(r,np.asarray(events),event_id=1,tmin=0,tmax=EPOCH_LEN_SEC-1/sf,baseline=None,preload=True,reject_by_annotation=True,metadata=pd.DataFrame(md),verbose=False)
                ep.pick_channels([c for c in COMMON_CHANNELS_14 if c in ep.ch_names],ordered=True)
                f=feature_rows(ep,'sky',s,cond,'experienced' if expmap.get(si)==1 else 'novice' if expmap.get(si)==0 else None,{'age':None,'artifact_detection':'skipped_fixed_threshold_invalid_for_SKY','bad_channels':','.join(bads),'ica_excluded':','.join(map(str,ics))})
                f['condition']=cond;f['epoch_idx']=ep.metadata.epoch_idx.to_numpy();f['t_start']=ep.metadata.t_start.to_numpy();out.append(f)
            print('SKY',s,'ok',flush=True)
        except Exception as e:
            import traceback; fail.append({'subject':s,'reason':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()});print('SKY',s,'FAILED',e,flush=True);traceback.print_exc()
        finally:gc.collect()
    return out,fail

def save(name,frames,failures):
    if frames:pd.concat(frames,ignore_index=True,sort=False).to_csv(OUT/f'{name}_all_subjects_features.csv',index=False)
    pd.DataFrame(failures,columns=['subject','reason','traceback']).to_csv(OUT/f'{name}_extraction_failures.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=None);ap.add_argument('--datasets',nargs='+',default=['braboszcz','brandmeyer','sky'],choices=['braboszcz','brandmeyer','sky']);ap.add_argument('--all-available',action='store_true',help='include subjects outside the existing feature-table analysis cohort');ap.add_argument('--sky-subjects',nargs='*',default=None);a=ap.parse_args()
    with threadpool_limits(limits=1):
        for ds,fn in [('braboszcz',run_braboszcz),('brandmeyer',run_brandmeyer),('sky',run_sky)]:
            if ds not in a.datasets:continue
            print('START',ds,'limit=',a.limit,flush=True);frames,fail=(fn(a.limit,all_available=a.all_available) if ds!='sky' else fn(a.limit,only_subjects=set(a.sky_subjects) if a.sky_subjects else None));save(ds,frames,fail)
            print('DONE',ds,'rows=',sum(map(len,frames)),'failures=',len(fail),flush=True)
if __name__=='__main__':
    import scipy.io as sio
    main()
