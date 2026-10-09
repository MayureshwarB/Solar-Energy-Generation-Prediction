# Methodology, assumptions, and limitations

## Data and time split

The combined workbook covers 2020-04-01 through 2024-03-31 (1,461 daily records). The scripts parse `DATE`, sort observations, set daily frequency, clean non-numeric values, convert generation from million units (MU) to kWh, and convert installed capacity from MW to kW where relevant.

The split is chronological:

- Training: 2020-04-01 through 2023-12-31
- Test: 2024-01-01 through 2024-03-31 (91 rows)

A chronological split is used because forecasting should predict dates later than the data used for training. Random splitting could let information from later dates influence the training set.

## Prophet reconstruction

Default features are the five weather variables specifically listed for Prophet in the project dissertation: `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `T2M`, `RH2M`, and `WS2M`.

- Target: solar generation in kWh
- Transformation: `log1p(kWh)`; predicted values are inverse-transformed with `expm1`
- Seasonality: multiplicative, yearly and weekly enabled, daily disabled
- `changepoint_prior_scale=0.05`
- `seasonality_prior_scale=10.0`

The original Prophet code and exact hyperparameter values were unavailable. The values above are explicit reconstruction choices and should not be described as confirmed original settings.

## SARIMAX reconstruction

A recovered code snippet specified `order=(1,1,1)`, `seasonal_order=(1,1,1,365)`, and external features, with training through 2023-12-31 and testing from 2024-01-01.

The exact seasonal configuration did not finish fitting within the available runtime. The saved SARIMAX scores therefore came from a faster approximation: explicitly difference the target and regressors by one day and 365 days, then fit `order=(1,0,1)`, `seasonal_order=(0,0,0,365)`. That retains ordinary ARMA terms and annual differencing while omitting seasonal AR and MA components. This is not the same model as the one in the recovered snippet.

## Missing data and evaluation

For Prophet, target gaps in the training period are interpolated. Originally missing test targets are not filled for scoring; metrics use originally observed test target values only. This avoids evaluating predictions against fabricated test labels. Feature interpolation is done within each split.

Metrics:

- MAE: mean absolute difference between actual and forecast values.
- RMSE: square root of mean squared error; gives larger errors more influence.
- MAPE: mean absolute percentage error; undefined/unstable for zero or near-zero actual values.
- R²: fit relative to predicting the test-set mean; it is not an accuracy percentage.

## Results are indicative, not an exact original reproduction

The saved reconstructed runs reported:

| Model/run | MAE (MU) | RMSE (MU) | MAPE | R² |
|---|---:|---:|---:|---:|
| SARIMAX fast approximation | 1.2332 | 1.6470 | 8.6733% | 0.3859 |
| Prophet reconstruction | 1.1421 | 1.4725 | 7.5732% | 0.5091 |

These values suggest Prophet performed better in these runs. However, because the SARIMAX run used an approximation and the model feature sets differ, they are not a controlled final comparison. The original report's numeric results table was incomplete, so these are newly reconstructed metrics.

Prophet uses weather regressors from the observed historical test-period workbook. This is a retrospective evaluation conditional on the weather data, not a real future forecast that would be available before the target dates. A deployed forecasting system would need predicted future weather features.

## Suggested next steps

1. Recover the original code and validate the exact settings if possible.
2. Add a seasonal-naive baseline.
3. Compare models on identical dates, target scales, observed labels, and feature availability.
4. Use rolling-origin validation rather than choosing on one test quarter only.
5. Test Fourier terms with ARIMA errors and a tree-based model such as XGBoost/LightGBM.
6. Build a Streamlit UI only after the prediction pipeline is reproducible.
