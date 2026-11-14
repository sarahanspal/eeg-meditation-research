"""Subject-level experienced/novice EEG analysis, stratified by dataset and state.
Uses harmonized re-extracted features under results/harmonized/.
"""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scipy.stats import false_discovery_control
from config import RAW_PATHS
from experience_mapping import build_ds3969_experience_map,build_sky_experience_map
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/harmonized'
META={'dataset','subject','condition','epoch_idx','experience_group','years_of_practice','age','group','style','bad_channels','ica_excluded','artifact_detection','session','probe_id','sub_epoch','t_start','label','y'}
SEED=20261111

def metadata_maps():
    br=pd.read_csv(Path(RAW_PATHS['braboszcz'])/'participants.tsv',sep='\t')
    b=br.set_index('participant_id')
    b_exp=build_ds3969_experience_map(RAW_PATHS['braboszcz'])
    bmroot=Path(RAW_PATHS['brandmeyer'])
    if not bmroot.exists():
        from decode_brandmeyer_probes import resolve_dir
        bmroot=resolve_dir(bmroot.parent,bmroot.name.strip())
    bm=pd.read_csv(bmroot/'participants.tsv',sep='\t').set_index('participant_id')
    skyroot=Path(RAW_PATHS['sky']);sky=pd.read_csv(skyroot/'SubjectMetaData.csv').set_index('S.No.')
    sexp=build_sky_experience_map(str(skyroot))
    maps={}
    for row in br.itertuples(index=False):
        sid=row.participant_id;num=int(sid.split('-')[1]);years=getattr(row,'years_of_practice',np.nan)
        maps[('braboszcz',sid)]=dict(experience='experienced' if b_exp.get(num)==1 else 'novice' if b_exp.get(num)==0 else None,
            years=float(years) if pd.notna(years) else np.nan,age=float(row.age) if pd.notna(row.age) else np.nan,group=row.group)
    for sid,row in bm.iterrows():
        maps[('brandmeyer',sid)]=dict(experience='experienced' if str(row.group).lower()=='expert' else 'novice',
            years=np.nan,age=float(row.age) if pd.notna(row.age) else np.nan,group=row.group)
    for sid,row in sky.iterrows():
        key=int(sid);sub=f'sky-{key:02d}';years=row['MeditationExperience(in years)']
        maps[('sky',sub)]=dict(experience='experienced' if sexp.get(key)==1 else 'novice' if sexp.get(key)==0 else None,
            years=float(years) if pd.notna(years) else np.nan,age=float(row.Age) if pd.notna(row.Age) else np.nan,group='SKY')
    return maps

def fdr_nullable(p):
    a=np.asarray(p,float);out=np.full(a.shape,np.nan);ok=np.isfinite(a)
    if ok.any():out[ok]=false_discovery_control(a[ok],method='bh')
    return out

def permutation_diff(X,g,n_perm=5000,seed=SEED):
    """Two-sided subject-label permutation p-values for each feature at once."""
    X=np.asarray(X,float);g=np.asarray(g,dtype=int);n1=int(g.sum());n0=len(g)-n1
    observed=X[g==1].mean(0)-X[g==0].mean(0)
    rng=np.random.default_rng(seed);hits=np.zeros(X.shape[1],int);done=0
    while done<n_perm:
        b=min(128,n_perm-done)
        order=np.argsort(rng.random((b,len(g))),axis=1)
        masks=np.zeros((b,len(g)),float);masks[np.arange(b)[:,None],order[:,:n1]]=1
        diffs=(masks@X)/n1-((1-masks)@X)/n0
        hits+=(np.abs(diffs)>=np.abs(observed)).sum(0);done+=b
    return observed,(hits+1)/(n_perm+1)

def main():
    maps=metadata_maps();frames=[]
    for ds in ('braboszcz','brandmeyer','sky'):
        p=OUT/f'{ds}_all_subjects_features.csv'
        d=pd.read_csv(p)
        # Attach authoritative subject-level covariates and corrected experience labels,
        # including the project's manually confirmed Braboszcz/SKY cases.
        d['experience_group']=[maps.get((ds,s),{}).get('experience') for s in d.subject]
        d['age']=[maps.get((ds,s),{}).get('age',np.nan) for s in d.subject]
        d['years_of_practice']=[maps.get((ds,s),{}).get('years',np.nan) for s in d.subject]
        d['group']=[maps.get((ds,s),{}).get('group') for s in d.subject]
        if ds=='braboszcz':
            style=d.group.astype('string').str.lower().map({'htr':'HYT','vip':'VIP','sny':'ISY'})
            d=d[style.notna()].copy();d['style']=style.loc[d.index]
        d.to_csv(p,index=False)
        for (sub,cond),g in d.groupby(['subject','condition']):
            meta=maps.get((ds,sub),{})
            if meta.get('experience') is None:continue
            feats=[c for c in g if c not in META]
            row=dict(dataset=ds,subject=sub,condition=cond,n_epochs=len(g),**meta)
            row.update(g[feats].mean(numeric_only=True).to_dict());frames.append(row)
    means=pd.DataFrame(frames);means.to_csv(OUT/'trait_subject_state_means.csv',index=False)
    feats=[c for c in means if c not in META|{'experience','years','group','n_epochs'}]
    cohort=means.groupby(['dataset','condition','experience']).subject.nunique().rename('n_subjects').reset_index()
    cohort.to_csv(OUT/'trait_state_cohort.csv',index=False)
    effects=[];corrs=[]
    for (ds,cond),g in means.groupby(['dataset','condition']):
        g=g.dropna(subset=['experience']); labels=g.experience.eq('experienced').astype(int).to_numpy()
        if labels.min()==labels.max():continue
        X=g[feats].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        # Small occasional feature gaps are filled by this state-level median after subject aggregation.
        med=np.nanmedian(X,axis=0);bad=~np.isfinite(X);X[bad]=np.take(med,np.where(bad)[1])
        diff,p=permutation_diff(X,labels,n_perm=5000,seed=SEED+len(effects))
        age=pd.to_numeric(g.age,errors='coerce').to_numpy(float); years=pd.to_numeric(g.years,errors='coerce').to_numpy(float)
        ageok=np.isfinite(age);yr_ok=np.isfinite(years)
        for j,f in enumerate(feats):
            a=X[labels==1,j];b=X[labels==0,j];sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(1,len(a)+len(b)-2))
            # Age-adjusted group coefficient: least squares on subject means, with HC-free small-sample t test.
            m=ageok & np.isfinite(X[:,j]);beta=np.nan;padj=np.nan
            if m.sum()>=6 and len(np.unique(labels[m]))==2:
                A=np.column_stack([np.ones(m.sum()),labels[m],age[m]]); yy=X[m,j];coef=np.linalg.lstsq(A,yy,rcond=None)[0];res=yy-A@coef;dfree=m.sum()-A.shape[1]
                if dfree>0:
                    cov=(res@res/dfree)*np.linalg.pinv(A.T@A);se=np.sqrt(max(cov[1,1],0));beta=coef[1]
                    if se>0:
                        from scipy.stats import t as student_t
                        padj=float(2*student_t.sf(abs(beta/se),dfree))
            effects.append(dict(dataset=ds,condition=cond,feature=f,n_experienced=int(labels.sum()),n_novice=int((1-labels).sum()),
                mean_experienced=a.mean(),mean_novice=b.mean(),difference=diff[j],cohens_d=(a.mean()-b.mean())/sp if sp>0 else np.nan,
                p_permutation=p[j],age_adjusted_difference=beta,p_age_adjusted=padj))
            if yr_ok.sum()>=6:
                rho,pc=spearmanr(years[yr_ok],X[yr_ok,j]);corrs.append(dict(dataset=ds,condition=cond,feature=f,n=int(yr_ok.sum()),spearman_rho=rho,p=pc))
    eff=pd.DataFrame(effects)
    if len(eff):
        eff['p_fdr_statewise']=eff.groupby(['dataset','condition']).p_permutation.transform(fdr_nullable)
        eff['p_age_fdr_statewise']=eff.groupby(['dataset','condition']).p_age_adjusted.transform(fdr_nullable)
        eff.to_csv(OUT/'trait_experience_feature_effects.csv',index=False)
    cor=pd.DataFrame(corrs)
    if len(cor):
        cor['p_fdr_statewise']=cor.groupby(['dataset','condition']).p.transform(fdr_nullable)
        cor.to_csv(OUT/'trait_years_practice_feature_correlations.csv',index=False)
    print(cohort.to_string(index=False));print('feature contrasts',len(eff),'years correlations',len(cor))
if __name__=='__main__':main()
