"""Chronological PM2.5 forecasting and validation-selected inference routing."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
import io,zipfile,json,time,platform,hashlib,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from threadpoolctl import threadpool_limits
threadpool_limits(1)
P=Path(__file__).resolve().parent
out=P/'results';out.mkdir(exist_ok=True)
archive=zipfile.ZipFile(P/'data/beijing.zip')
frames=[]
def ingest(z):
    for name in z.namelist():
        if name.endswith('.zip'): ingest(zipfile.ZipFile(io.BytesIO(z.read(name))))
        elif name.endswith('.csv') and 'PRSA_Data_' in name:
            frames.append(pd.read_csv(z.open(name)))
ingest(archive)
df=pd.concat(frames,ignore_index=True)
df['date']=pd.to_datetime(df[['year','month','day','hour']])
df=df.sort_values(['station','date']).reset_index(drop=True)
g=df.groupby('station',sort=False)
df['target']=g['PM2.5'].shift(-1)
df['target_date']=g['date'].shift(-1)
df['lag1']=g['PM2.5'].shift(1);df['lag2']=g['PM2.5'].shift(2)
df['lag24']=g['PM2.5'].shift(24)
df['change']=abs(df['PM2.5']-df['lag1'])
df['hour_sin']=np.sin(2*np.pi*df.hour/24);df['hour_cos']=np.cos(2*np.pi*df.hour/24)
df['month_sin']=np.sin(2*np.pi*df.month/12);df['month_cos']=np.cos(2*np.pi*df.month/12)
# Keep missing predictors for train-fitted imputation; require observed current PM and future target.
df=df[df['PM2.5'].notna()&df.target.notna()&((df.target_date-df.date)==pd.Timedelta(hours=1))].copy()
cols=['PM2.5','lag1','lag2','lag24','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','hour_sin','hour_cos','month_sin','month_cos']
X=pd.concat([df[cols],pd.get_dummies(df.station,prefix='station',dtype=float),pd.get_dummies(df.wd,prefix='wind',dtype=float)],axis=1).astype(float)
# Apply boundaries to target timestamp to prevent targets crossing partitions.
masks=[df.target_date<pd.Timestamp('2015-01-01'),(df.target_date>=pd.Timestamp('2015-01-01'))&(df.target_date<pd.Timestamp('2016-01-01')),df.target_date>=pd.Timestamp('2016-01-01')]
xs=[X.loc[m].to_numpy() for m in masks];ys=[df.loc[m,'target'].to_numpy() for m in masks]
meta={'raw_rows':sum(len(x) for x in frames),'eligible_rows':len(df),'features':X.columns.tolist(),'split_counts':[len(y) for y in ys],'stations':sorted(df.station.unique().tolist()),'python':platform.python_version(),'sklearn':sklearn.__version__,'cpu':platform.processor(),'platform':platform.platform(),'data_sha256':hashlib.sha256((P/'data/beijing.zip').read_bytes()).hexdigest()}
small=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0))
large=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=31,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=42)
for name,model in [('ridge',small),('boosting',large)]:
    t=time.perf_counter();model.fit(xs[0],ys[0]);meta[name+'_training_s']=time.perf_counter()-t
    meta[name+'_pickle_bytes']=len(pickle.dumps(model))
pv=small.predict(xs[1]);lv=large.predict(xs[1]);pt=small.predict(xs[2]);lt=large.predict(xs[2])
def metric(y,p):return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5)}
rows=[]; predictions={}
def add(name,p,rate):
    rows.append(dict(method=name,heavy_fraction=float(rate),**metric(ys[2],p)));predictions[name]=p
add('Persistence',df.loc[masks[2],'PM2.5'].to_numpy(),0)
add('Ridge',pt,0);add('Always boosting',lt,1)
valbase=mean_absolute_error(ys[1],lv)
# Input-only volatility routing, thresholds chosen on validation only.
sv=df.loc[masks[1],'change'].fillna(np.inf).to_numpy();st=df.loc[masks[2],'change'].fillna(np.inf).to_numpy()
thresholds=np.unique(np.r_[-np.inf,0,np.quantile(sv[np.isfinite(sv)],np.linspace(0,1,101)),np.inf])
validation=[]
for th in thresholds:
    use=sv>th
    validation.append({'threshold':float(th),'heavy_fraction':float(use.mean()),'mae':float(mean_absolute_error(ys[1],np.where(use,lv,pv)))})
pd.DataFrame(validation).to_csv(out/'validation_sweep.csv',index=False)
for tolerance in [0,.02,.05,.10]:
    feasible=[r for r in validation if r['mae']<=valbase*(1+tolerance)]
    best=min(feasible,key=lambda r:(r['heavy_fraction'],r['mae']))
    use=st>best['threshold'];pred=np.where(use,lt,pt)
    name=f'Volatility {int(tolerance*100)}pct';add(name,pred,use.mean())
    rows[-1].update(threshold=best['threshold'],validation_mae=best['mae'],validation_heavy_fraction=best['heavy_fraction'])
    # Random routing matched to observed test rate; repeated seeds do not retrain models.
    rand_mae=[]
    for seed in range(30):
        rng=np.random.default_rng(seed);ru=rng.random(len(pt))<use.mean()
        rand_mae.append(mean_absolute_error(ys[2],np.where(ru,lt,pt)))
    rows[-1]['random_matched_mae_mean']=float(np.mean(rand_mae));rows[-1]['random_matched_mae_sd']=float(np.std(rand_mae,ddof=1))
res=pd.DataFrame(rows)
# True routed batch calls, including ridge and mask overhead; 9 repeats after one warmup.
def routed(x,score,th):
    p=small.predict(x);m=score>th
    if m.any():p[m]=large.predict(x[m])
    return p
for i,row in res.iterrows():
    def predict():
        if row.method=='Persistence':return df.loc[masks[2],'PM2.5'].to_numpy()
        if row.method=='Ridge':return small.predict(xs[2])
        if row.method=='Always boosting':return large.predict(xs[2])
        return routed(xs[2],st,row.threshold)
    predict();times=[]
    for _ in range(9):
        t=time.perf_counter();p=predict();times.append(time.perf_counter()-t)
    assert np.allclose(p,predictions[row.method])
    res.loc[i,'batch_ms_median']=np.median(times)*1000
    res.loc[i,'batch_ms_q25']=np.quantile(times,.25)*1000
    res.loc[i,'batch_ms_q75']=np.quantile(times,.75)*1000
# Station-level paired errors and daily moving-block-independent bootstrap across calendar days.
test=df.loc[masks[2],['date','station','target']].reset_index(drop=True)
for name,p in predictions.items():test[name]=p
station=[]
for station_name,part in test.groupby('station'):
    for name in predictions:station.append({'station':station_name,'method':name,**metric(part.target,part[name])})
pd.DataFrame(station).to_csv(out/'station_metrics.csv',index=False)
boot={}
days=test.date.dt.floor('D')
for name in predictions:
    err=np.abs(test[name]-test.target)-np.abs(test['Always boosting']-test.target)
    daily=pd.DataFrame({'day':days,'sum':err,'n':1}).groupby('day').sum()
    rng=np.random.default_rng(2026);vals=[]
    for _ in range(2000):
        idx=rng.integers(0,len(daily),len(daily));a=daily.iloc[idx];vals.append(a['sum'].sum()/a.n.sum())
    boot[name]={'mae_delta':float(err.mean()),'daily_cluster_bootstrap_95ci':np.quantile(vals,[.025,.975]).tolist()}
test.to_csv(out/'test_predictions.csv.gz',index=False)
res.to_csv(out/'metrics.csv',index=False)
meta['validation_always_boosting_mae']=valbase;meta['bootstrap']=boot
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print(res.to_string(index=False));print(json.dumps(meta,indent=2))
