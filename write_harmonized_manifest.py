"""Write reproducibility metadata for the completed harmonized extraction."""
import hashlib,json,platform,sys
from pathlib import Path
import importlib.metadata as md
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/harmonized'
FILES=[OUT/f'{d}_all_subjects_features.csv' for d in ('braboszcz','brandmeyer','sky')]
ANALYSIS_FILES=[
 ROOT/'results/combined_loso_summary.csv',ROOT/'results/combined_loso_statistics.csv',
 ROOT/'results/combined_dataset_probe.csv',ROOT/'results/combined_blocked_controls.csv',
 ROOT/'results/combined_learning_curve_summary.csv',ROOT/'results/combined_feature_sign_consistency.csv',
 OUT/'calibration_decomposition_subject.csv',OUT/'calibration_null_adjusted_gain.csv',
 OUT/'feature_specification_summary.csv',OUT/'trait_state_cohort.csv',
 OUT/'trait_experience_feature_effects.csv',OUT/'trait_years_practice_feature_correlations.csv',
 OUT/'braboszcz_classifier_cohort.csv',OUT/'braboszcz_style_state_loso_summary.csv',
 OUT/'braboszcz_med_style_loso_macro_summary.csv',OUT/'braboszcz_style_classifier_statistics.csv',
 OUT/'csp_matched_loso_subject.csv',OUT/'csp_harmonized_loso_subject.csv',OUT/'cohort_by_analysis.csv',
 OUT/'cohort_by_analysis_exclusions.csv',OUT/'braboszcz_run_order.csv',OUT/'numerical_warning_feature_profile.csv',
 OUT/'numerical_warning_blas_comparison.log',OUT/'research_summary.png',
 OUT/'trait_ctr_vs_experienced.csv',OUT/'dose_response_years.csv',OUT/'riemann_loso_subject.csv',OUT/'riemann_vs_feature_paired.csv',
 OUT/'preproc_sensitivity.csv',OUT/'aperiodic_sign_consistency.csv',OUT/'braboszcz_ctr_features.csv',OUT/'brandmeyer_probe_fold_check.csv',
 OUT/'brandmeyer_ratings_by_group.csv',OUT/'calibration_curve_summary.csv',ROOT/'results/paper_numbers.csv',ROOT/'results/paper_numbers_notes.csv',
 ROOT/'results/methods_facts.md',ROOT/'results/methods_facts_per_subject.csv',
 ROOT/'results/figures/fig1_accuracy_ladder.pdf',ROOT/'results/figures/fig2_theta_sign_consistency.pdf',ROOT/'results/figures/fig3_calibration.pdf']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 outputs={}
 for p in FILES:
  if not p.exists():continue
  d=pd.read_csv(p);meta={'dataset':p.stem.replace('_all_subjects_features',''),'n_rows':len(d),'n_subjects':int(d.subject.nunique()),'conditions':d.groupby('condition').size().to_dict(),'sha256':sha(p)}
  feats=[c for c in d if c.endswith(('_delta','_theta','_alpha','_beta','_gamma1','_gamma2','_IAF')) or '_ratio_' in c or c.startswith('asym_')]
  meta['n_feature_columns']=len(feats);meta['has_nonfinite']=bool((~np.isfinite(d[feats].to_numpy(float))).any())
  outputs[p.name]=meta
 manifest={'python':sys.version,'platform':platform.platform(),'packages':{p:md.version(p) for p in ('numpy','pandas','scipy','scikit-learn','mne','pyriemann','fooof','matplotlib')},
  'preprocessing':{'channels':['C3','C4','F3','F4','F7','F8','O1','O2','P3','P4','P7','P8','T7','T8'],'reference':'common-channel average reference','notch_hz':[50,60],'highpass_hz':1.0,'highpass_iir':'6th-order Butterworth, zero phase','sampling_rate_hz':256,'epoch_seconds':4,'epoch_overlap_seconds':0,'ica':'Infomax extended, random_state=97, shared across runs within subject, fit decim=2','feature_schema':'canonical bands, IAF, asymmetry, cross-band ratios'},
 'superseded':{'csp_matched_loso_subject.csv':'Legacy-array CSP (med1breath+med2 vs think1+think2, different preprocessing); superseded by csp_harmonized_loso_subject.csv. Kept, not for paper numbers.'},
 'known_warnings':{'sklearn_matmul_overflow_invalid':'Spurious FP flags from numpy 2.0.2 + Apple Accelerate (macOS 15); features finite, |x|<=13, identical balanced accuracy under numpy 1.26.4/OpenBLAS (0 warnings). See numerical_warning_blas_comparison.log.'},
 'dataset_exceptions':{'sky':'The fixed dB bad-segment detector flagged >=95% of recording; annotation rejection was skipped for SKY. SKY inputs retain 4-second timed epochs and use common filters/reference/ICA.',
   'braboszcz_classifier_cohort':'Classifier analyses include HYT, VIP, ISY only; equal n=18 subjects/style, selected with seed 20261111. CTR and TM excluded. The full harmonized feature table remains unbalanced (18 HYT, 18 VIP, 20 ISY) for descriptive/trait summaries.'},
 'code_sha256':{n:sha(ROOT/n) for n in ('reextract_harmonized.py','braboszcz_classifier_cohort.py','braboszcz_style_classifiers.py','braboszcz_style_statistics.py','csp_harness.py','combined_analysis.py','combined_supplementary.py','trait_experience_analysis.py','calibration_decomposition.py','feature_specification_curve.py','cohort_reconciliation.py','csp_harmonized_epochs.py','csp_harmonized_loso.py','cohort_by_analysis.py','braboszcz_run_order.py','numerical_warning_diagnosis_probe.py','numerical_warning_blas_comparison.py','extract_braboszcz_ctr.py','trait_ctr_vs_experienced.py','dose_response_years.py','riemann_loso.py','preproc_variants.py','preproc_sensitivity.py','aperiodic_theta.py','brandmeyer_sanity.py','calibration_curve.py','paper_numbers.py','methods_facts.py','make_figures.py','write_harmonized_manifest.py','plot_research_summary.py') if (ROOT/n).exists()},
  'analysis_outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in ANALYSIS_FILES if p.exists()},
  'outputs':outputs}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
 print(OUT/'manifest.json')
if __name__=='__main__':main()
