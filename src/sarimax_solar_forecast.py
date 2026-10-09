import pandas as pd, numpy as np, os, warnings, time, json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path='/mnt/data/Complete Data 2020 - 2024 (all features).xlsx'
df=pd.read_excel(path,parse_dates=['DATE']).set_index('DATE').asfreq('D')
df.columns=df.columns.str.strip(); df.replace(['NA','NaN','na','n/a','-',' '],np.nan,inplace=True)
missing_before=int(df.isna().sum().sum()); missing_target_before=int(df['Solar Power Generated (MU)'].isna().sum())
df.interpolate(method='time',inplace=True)
df['Installed Capacity (kW)']=df['Installed Capacity (MW)']*1000
df['Solar Power Generated (kWh)']=df['Solar Power Generated (MU)']*1_000_000
df.drop(columns=['Installed Capacity (MW)','Solar Power Generated (MU)'],inplace=True)
df=df.round(3)
target='Solar Power Generated (kWh)'; exog_vars=[c for c in df.columns if c!=target]
train_end=pd.Timestamp('2023-12-31'); test_start=pd.Timestamp('2024-01-01')
# Explicitly apply (1-B)(1-B^365) differencing to target and every regressor so test-period regressors are transformed correctly.
all_transformed=df.diff(365).diff(1)
# train on transformed rows whose dates are no later than train_end; test rows remain chronological
train=all_transformed.loc[:train_end].dropna()
test=all_transformed.loc[test_start:]
# remove rows with invalid features if any remain
train=train.dropna(subset=[target]+exog_vars); test=test.dropna(subset=[target]+exog_vars)
print(f'Dataset rows={len(df)}; dates={df.index.min().date()} to {df.index.max().date()}; missing entries before interpolation={missing_before}, target missing={missing_target_before}')
print(f'Transformed training rows={len(train)}; first={train.index.min().date()}, last={train.index.max().date()}')
print(f'Transformed test rows={len(test)}; first={test.index.min().date()}, last={test.index.max().date()}')
print('Exogenous features:',exog_vars)
# This is a computationally practical variant: keep nonseasonal ARMA(1,1) and annual differencing,
# but omit seasonal AR and MA terms which make the exact 365-period state space very large.
t0=time.time()
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 model=SARIMAX(train[target], exog=train[exog_vars], order=(1,0,1), seasonal_order=(0,0,0,365), enforce_stationarity=False, enforce_invertibility=False)
 results=model.fit(disp=False,maxiter=100,method='lbfgs')
 print('Fit time seconds:',round(time.time()-t0,2),'Converged:',results.mle_retvals.get('converged'),'iterations:',results.mle_retvals.get('iterations'))
 print('Warnings:',[str(w.message) for w in caught[-8:]])
# Forecast transformed target and invert z_t = (1-B)(1-B^365)y_t:
# y_t = z_t + y_(t-1) + y_(t-365) - y_(t-366).
z_pred=np.asarray(results.get_forecast(steps=len(test), exog=test[exog_vars]).predicted_mean,dtype=float)
raw=np.asarray(df[target],dtype=float); all_idx=df.index; pred_pos=[all_idx.get_loc(dt) for dt in test.index]
yhat=[]
for pos,z in zip(pred_pos,z_pred):
    prev=yhat[-1] if yhat else raw[pos-1]
    y_val=z+prev+raw[pos-365]-raw[pos-366]
    yhat.append(y_val)
forecast=pd.Series(yhat,index=test.index,name='Predicted generation (kWh)')
actual=df.loc[test.index,target].astype(float)
mae=float(mean_absolute_error(actual,forecast)); rmse=float(np.sqrt(mean_squared_error(actual,forecast))); mape=float(mean_absolute_percentage_error(actual,forecast)*100); r2=float(r2_score(actual,forecast))
metrics={'MAE_kWh':mae,'RMSE_kWh':rmse,'MAPE_percent':mape,'R2':r2}
print('METRICS:',json.dumps(metrics,indent=2))
print('AIC:',results.aic,'BIC:',results.bic)
print('Actual MU stats:',(actual/1e6).describe().to_dict())
print('Predicted MU stats:',(forecast/1e6).describe().to_dict())
# metrics plot - MAE/RMSE shown in MU to keep scale legible
pd.DataFrame({'Metric':['MAE (MU)','RMSE (MU)','MAPE (%)','R²'],'Value':[mae/1e6,rmse/1e6,mape,r2]}).to_csv('/mnt/data/sarimax_metrics.csv',index=False)
pd.DataFrame({'Actual Generation (kWh)':actual,'Predicted Generation (kWh)':forecast,'Error (Actual - Predicted) (kWh)':actual-forecast,'Absolute Error (kWh)':np.abs(actual-forecast)}).to_csv('/mnt/data/sarimax_forecast_results.csv',index_label='DATE')
plt.figure(figsize=(14,6)); plt.plot(actual.index,actual/1e6,label='Actual',linewidth=2); plt.plot(forecast.index,forecast/1e6,label='Forecast',linestyle='--',linewidth=2); plt.xlabel('Date'); plt.ylabel('Daily generation (MU)'); plt.title('Fast SARIMAX variant — annual differencing; seasonal AR/MA omitted'); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig('/mnt/data/sarimax_forecast_plot.png',dpi=180); plt.close()
plt.figure(figsize=(14,4.5)); plt.axhline(0,color='black',linewidth=1); plt.plot(actual.index,(actual-forecast)/1e6); plt.xlabel('Date'); plt.ylabel('Error (MU)'); plt.title('Forecast errors — fast SARIMAX variant (actual − forecast)'); plt.grid(True,alpha=.25); plt.tight_layout(); plt.savefig('/mnt/data/sarimax_error_plot.png',dpi=180); plt.close()
plt.figure(figsize=(6.5,6)); plt.scatter(actual/1e6,forecast/1e6,alpha=.8); lo=min((actual/1e6).min(),(forecast/1e6).min()); hi=max((actual/1e6).max(),(forecast/1e6).max()); plt.plot([lo,hi],[lo,hi],linestyle='--',label='Perfect prediction'); plt.xlabel('Actual generation (MU)'); plt.ylabel('Predicted generation (MU)'); plt.title('Actual vs predicted — fast SARIMAX variant'); plt.legend(); plt.grid(True,alpha=.25); plt.tight_layout(); plt.savefig('/mnt/data/sarimax_actual_vs_predicted.png',dpi=180); plt.close()
plt.figure(figsize=(8,4.8)); labels=['MAE (MU)','RMSE (MU)','MAPE (%)','R²']; vals=[mae/1e6,rmse/1e6,mape,r2]; bars=plt.bar(labels,vals); plt.title('Forecast evaluation metrics (different units; read labels)'); plt.ylabel('Value'); plt.grid(axis='y',alpha=.25); plt.tight_layout(); plt.savefig('/mnt/data/sarimax_metrics_plot.png',dpi=180); plt.close()
with open('/mnt/data/sarimax_run_summary.txt','w') as f:
 f.write('Solar energy forecast run summary\n\n')
 f.write(f'Workbook: {os.path.basename(path)}\nDates: {df.index.min().date()} to {df.index.max().date()} ({len(df)} daily records)\n')
 f.write(f'Missing entries before interpolation: {missing_before}; missing target values: {missing_target_before}\n')
 f.write(f'Train cutoff: {train_end.date()}\nTest dates: {test.index.min().date()} to {test.index.max().date()} ({len(test)} daily rows)\n')
 f.write('Original code target model: order=(1,1,1), seasonal_order=(1,1,1,365), 11 exogenous features. Exact fit did not finish in 4 minutes here; its expanded state space had 733 states and one likelihood evaluation alone exceeded 120 seconds without simple differencing.\n')
 f.write('Computed model is a fast approximation: explicitly difference target and all exogenous features by 1 day and 365 days, then fit SARIMAX order=(1,0,1), seasonal_order=(0,0,0,365). This retains ordinary ARMA terms and annual differencing but omits seasonal AR and MA components. Forecasts inverted to kWh. Therefore these metrics are NOT the exact scores of the original seasonal_order=(1,1,1,365) model.\n')
 f.write(f'Converged: {results.mle_retvals.get("converged")}\nIterations: {results.mle_retvals.get("iterations")}\nAIC: {results.aic}\nBIC: {results.bic}\n')
 f.write(f'MAE (kWh): {mae}\nRMSE (kWh): {rmse}\nMAPE (%): {mape}\nR2: {r2}\n')
 f.write('MAE/RMSE are from prediction on the held-out 2024-01-01 to 2024-03-31 period. MAPE and R² were additionally calculated; the provided code only calculates MAE and RMSE.\n')
