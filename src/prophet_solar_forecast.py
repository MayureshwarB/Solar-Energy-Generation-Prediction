#!/usr/bin/env python3
"""Reconstruct the Solar Energy Generation Prediction Prophet experiment.

This script requires the `prophet` Python package and works with the uploaded
Complete Data 2020 - 2024 (all features).xlsx workbook.

It implements the report-described choices where they are specified:
- daily date index
- target converted from MU to kWh
- installed capacity converted from MW to kW
- time-based interpolation for missing training targets and feature values
- logarithmic target transform
- Prophet multiplicative seasonality
- five weather regressors explicitly listed in the dissertation by default (optional --feature-set all)
- chronological train/test split: training through 2023-12-31; test from 2024-01-01
- MAE, RMSE, MAPE, R-squared, CSVs and plots

Important: the original code is unavailable, and the report does not preserve exact
Prophet hyperparameters or numerical metrics. This is a transparent reconstruction,
not a guaranteed reproduction of the original run. It defaults to the five weather
features listed for Prophet in the dissertation; `--feature-set all` uses every available
weather/capacity column.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

TARGET_MU = "Solar Power Generated (MU)"
TARGET_KWH = "Solar Power Generated (kWh)"
CAPACITY_MW = "Installed Capacity (MW)"
CAPACITY_KW = "Installed Capacity (kW)"
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
# The dissertation explicitly lists these five weather regressors for the Prophet model.
REPORT_PROPHET_FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "WS2M",
]


def metric_values(actual_kwh: np.ndarray, predicted_kwh: np.ndarray) -> dict[str, float]:
    actual_kwh = np.asarray(actual_kwh, dtype=float)
    predicted_kwh = np.asarray(predicted_kwh, dtype=float)
    mask = np.isfinite(actual_kwh) & np.isfinite(predicted_kwh)
    actual_kwh, predicted_kwh = actual_kwh[mask], predicted_kwh[mask]
    if len(actual_kwh) == 0:
        raise ValueError("No observed test target values are available for scoring.")
    mae = float(mean_absolute_error(actual_kwh, predicted_kwh))
    rmse = float(np.sqrt(mean_squared_error(actual_kwh, predicted_kwh)))
    nonzero = np.abs(actual_kwh) > 1e-12
    mape = float(np.mean(np.abs((actual_kwh[nonzero] - predicted_kwh[nonzero]) / actual_kwh[nonzero])) * 100) if np.any(nonzero) else float("nan")
    r2 = float(r2_score(actual_kwh, predicted_kwh)) if len(actual_kwh) >= 2 else float("nan")
    return {
        "n_scored_test_observations": int(len(actual_kwh)),
        "MAE_kWh": mae,
        "MAE_MU": mae / 1_000_000,
        "RMSE_kWh": rmse,
        "RMSE_MU": rmse / 1_000_000,
        "MAPE_percent": mape,
        "R2": r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Prophet for daily solar generation.")
    parser.add_argument("--input", required=True, help="Path to combined Excel workbook")
    parser.add_argument("--output-dir", default="prophet_solar_output", help="Directory for outputs")
    parser.add_argument(
        "--feature-set", choices=["report", "all"], default="report",
        help="'report' uses the five weather regressors explicitly listed in the dissertation; 'all' uses all available weather/capacity columns."
    )
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Excel workbook not found: {input_path}")

    # Load, normalize column names, and preserve which target values were truly observed.
    df = pd.read_excel(input_path, parse_dates=["DATE"])
    df.columns = df.columns.astype(str).str.strip()
    if "DATE" not in df.columns:
        raise ValueError("Expected a DATE column in the workbook.")
    if TARGET_MU not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_MU}'. Columns found: {df.columns.tolist()}")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").drop_duplicates(subset="DATE", keep="last")
    df = df.set_index("DATE").asfreq("D")

    # Convert invalid textual values to NaN and make data numeric.
    df = df.replace(["NA", "NaN", "na", "n/a", "-", " "], np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw_target_mu = df[TARGET_MU].copy()
    observed_target = raw_target_mu.notna()
    df[TARGET_KWH] = raw_target_mu * 1_000_000
    df[CAPACITY_KW] = df[CAPACITY_MW] * 1_000 if CAPACITY_MW in df.columns else np.nan
    for col in [TARGET_MU, CAPACITY_MW]:
        if col in df.columns:
            df = df.drop(columns=col)
    # Keep the mask before filling to avoid scoring against imputed test labels.
    df["_target_observed"] = observed_target
    all_feature_cols = [c for c in df.columns if c not in [TARGET_KWH, "_target_observed"]]
    if args.feature_set == "report":
        missing_report_features = [c for c in REPORT_PROPHET_FEATURES if c not in all_feature_cols]
        if missing_report_features:
            raise ValueError(f"Workbook is missing Prophet features listed in the dissertation: {missing_report_features}")
        feature_cols = REPORT_PROPHET_FEATURES.copy()
    else:
        feature_cols = all_feature_cols
    if not feature_cols:
        raise ValueError("No weather/capacity regressors found after preprocessing.")

    # Split chronologically first. Interpolate regressors independently within each split
    # to avoid using later test data to fill training-period feature gaps.
    train_mask = df.index <= TRAIN_END
    test_mask = df.index >= TEST_START
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Data does not contain both the expected training and test periods.")

    for cols_mask in [train_mask, test_mask]:
        sub = df.loc[cols_mask, feature_cols].copy()
        sub = sub.interpolate(method="time", limit_direction="both").ffill().bfill()
        df.loc[cols_mask, feature_cols] = sub

    # Target is interpolated in training only. Actual test targets remain unfilled.
    train_target = df.loc[train_mask, TARGET_KWH].copy()
    train_target_filled = train_target.interpolate(method="time", limit_direction="both").ffill().bfill()
    train_imputed_count = int(train_target.isna().sum())
    if train_target_filled.isna().any():
        raise ValueError("Training target still contains missing values after interpolation.")
    if (train_target_filled <= 0).any():
        raise ValueError("Target contains non-positive training generation; log1p target transform expected.")

    # Report-described log transform and multiplicative Prophet model.
    train_prophet = pd.DataFrame({"ds": df.index[train_mask], "y": np.log1p(train_target_filled.values)})
    for col in feature_cols:
        train_prophet[col] = df.loc[train_mask, col].values
    test_prophet = pd.DataFrame({"ds": df.index[test_mask]})
    for col in feature_cols:
        test_prophet[col] = df.loc[test_mask, col].values

    model = Prophet(
        growth="linear",
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    for col in feature_cols:
        model.add_regressor(col, standardize="auto")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train_prophet)
    forecast_log = model.predict(test_prophet)
    # Inverse-transform forecasts to kWh. The target used log1p, so use expm1.
    forecast_kwh = np.expm1(forecast_log["yhat"].to_numpy())
    lower_kwh = np.expm1(forecast_log["yhat_lower"].to_numpy())
    upper_kwh = np.expm1(forecast_log["yhat_upper"].to_numpy())

    test_index = df.index[test_mask]
    actual_kwh_all = df.loc[test_mask, TARGET_KWH].to_numpy(dtype=float)  # originally missing values remain NaN
    actual_observed_test = df.loc[test_mask, "_target_observed"].to_numpy(dtype=bool)
    scored = actual_observed_test & np.isfinite(actual_kwh_all) & np.isfinite(forecast_kwh)
    metrics = metric_values(actual_kwh_all[scored], forecast_kwh[scored])

    results = pd.DataFrame({
        "DATE": test_index,
        "Actual_Observed": actual_observed_test,
        "Actual_Generation_kWh": actual_kwh_all,
        "Actual_Generation_MU": actual_kwh_all / 1_000_000,
        "Prophet_Predicted_kWh": forecast_kwh,
        "Prophet_Predicted_MU": forecast_kwh / 1_000_000,
        "Prediction_Lower_95pct_kWh": lower_kwh,
        "Prediction_Upper_95pct_kWh": upper_kwh,
        "Prediction_Lower_95pct_MU": lower_kwh / 1_000_000,
        "Prediction_Upper_95pct_MU": upper_kwh / 1_000_000,
        "Error_kWh_actual_minus_predicted": actual_kwh_all - forecast_kwh,
        "Error_MU_actual_minus_predicted": (actual_kwh_all - forecast_kwh) / 1_000_000,
        "Absolute_Percent_Error": np.where(
            np.isfinite(actual_kwh_all) & (np.abs(actual_kwh_all) > 1e-12),
            np.abs(actual_kwh_all - forecast_kwh) / np.abs(actual_kwh_all) * 100,
            np.nan,
        ),
    })
    results.to_csv(outdir / "prophet_forecast_results.csv", index=False, date_format="%Y-%m-%d")

    metrics_df = pd.DataFrame([
        {"Metric": "MAE", "Value": metrics["MAE_kWh"], "Unit": "kWh", "Interpretation": "Mean absolute error"},
        {"Metric": "MAE", "Value": metrics["MAE_MU"], "Unit": "MU", "Interpretation": "Mean absolute error"},
        {"Metric": "RMSE", "Value": metrics["RMSE_kWh"], "Unit": "kWh", "Interpretation": "Root mean squared error"},
        {"Metric": "RMSE", "Value": metrics["RMSE_MU"], "Unit": "MU", "Interpretation": "Root mean squared error"},
        {"Metric": "MAPE", "Value": metrics["MAPE_percent"], "Unit": "%", "Interpretation": "Mean absolute percentage error; only defined for non-zero actuals"},
        {"Metric": "R2", "Value": metrics["R2"], "Unit": "score", "Interpretation": "Coefficient of determination; may be negative"},
    ])
    metrics_df.to_csv(outdir / "prophet_metrics.csv", index=False)
    with open(outdir / "prophet_metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write("Solar Energy Generation Prediction — Prophet reconstruction\n")
        f.write("=========================================================\n\n")
        f.write(f"Input workbook: {input_path.name}\n")
        f.write(f"Date coverage: {df.index.min().date()} to {df.index.max().date()}\n")
        f.write(f"Training: {df.index[train_mask].min().date()} to {df.index[train_mask].max().date()}\n")
        f.write(f"Test: {df.index[test_mask].min().date()} to {df.index[test_mask].max().date()}\n")
        f.write(f"Training observations: {int(train_mask.sum())}\n")
        f.write(f"Test observations available: {int(test_mask.sum())}\n")
        f.write(f"Observed test targets scored: {metrics['n_scored_test_observations']}\n")
        f.write(f"Target values interpolated within training period: {train_imputed_count}\n")
        f.write(f"Log transform: log1p(kWh); predictions inverse transformed with expm1\n")
        f.write(f"Seasonality mode: multiplicative; yearly=True; weekly=True; daily=False\n")
        f.write(f"Changpoint prior scale: 0.05; seasonality prior scale: 10.0\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write("External regressors:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")
        f.write("\nMetrics on originally observed test targets only:\n")
        f.write(f"MAE: {metrics['MAE_kWh']:.4f} kWh ({metrics['MAE_MU']:.6f} MU)\n")
        f.write(f"RMSE: {metrics['RMSE_kWh']:.4f} kWh ({metrics['RMSE_MU']:.6f} MU)\n")
        f.write(f"MAPE: {metrics['MAPE_percent']:.4f}%\n")
        f.write(f"R2: {metrics['R2']:.6f}\n\n")
        f.write("Interpretation and caveats:\n")
        f.write("- Metrics were recomputed from this reconstructed run; original code/scores were unavailable.\n")
        f.write("- Metrics are evaluated against observed test targets only; imputed test targets are excluded.\n")
        f.write("- Test-period weather/capacity regressors are taken from the supplied historical data. This is a retrospective conditional evaluation, not a live future-weather forecast.\n")
        f.write("- MAE/RMSE depend on the target units; do not call R2 an accuracy percentage.\n")
        if caught:
            f.write("\nWarnings recorded during model fit:\n")
            for w in caught[:30]:
                f.write(f"- {w.message}\n")

    # Plot 1: test actual vs forecast, units in MU for easier reading.
    fig, ax = plt.subplots(figsize=(14, 6))
    actual_plot = np.where(actual_observed_test, actual_kwh_all / 1_000_000, np.nan)
    ax.plot(test_index, actual_plot, label="Actual generation (observed)", linewidth=2)
    ax.plot(test_index, forecast_kwh / 1_000_000, label="Prophet forecast", linestyle="--", linewidth=2)
    ax.fill_between(test_index, lower_kwh / 1_000_000, upper_kwh / 1_000_000,
                    alpha=0.18, label="95% forecast interval")
    ax.set_title("Solar Energy Generation: Prophet Actual vs Forecast (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy generated (MU = million kWh)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prophet_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: residuals over time, observed targets only.
    residual = (actual_kwh_all[scored] - forecast_kwh[scored]) / 1_000_000
    scored_dates = test_index[scored]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.plot(scored_dates, residual, marker="o", markersize=3, linewidth=1.2)
    ax.set_title("Prophet Forecast Residuals (Actual − Predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Error (MU)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "prophet_error_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: actual vs predicted scatter with ideal diagonal.
    actual_mu = actual_kwh_all[scored] / 1_000_000
    predicted_mu = forecast_kwh[scored] / 1_000_000
    low = float(min(actual_mu.min(), predicted_mu.min()))
    high = float(max(actual_mu.max(), predicted_mu.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual_mu, predicted_mu, alpha=0.8)
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5, label="Perfect predictions")
    ax.set_title("Prophet: Actual vs Predicted Generation")
    ax.set_xlabel("Actual generation (MU)")
    ax.set_ylabel("Predicted generation (MU)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prophet_actual_vs_predicted_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: metrics in four appropriately labelled panels.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0, 0], "MAE (MU)", metrics["MAE_MU"], "Lower is better"),
        (axes[0, 1], "RMSE (MU)", metrics["RMSE_MU"], "Lower is better"),
        (axes[1, 0], "MAPE (%)", metrics["MAPE_percent"], "Lower is better"),
        (axes[1, 1], "R²", metrics["R2"], "Higher is generally better"),
    ]
    for ax, title, val, direction in panels:
        ax.bar([title], [val])
        ax.set_title(title + "\n" + direction)
        ax.text(0, val, f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Prophet Evaluation Metrics — Observed Test Targets")
    fig.tight_layout()
    fig.savefig(outdir / "prophet_metrics_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: full observed history, train/test boundary, and test-period forecasts.
    fig, ax = plt.subplots(figsize=(14, 6))
    history_mu = df[TARGET_KWH] / 1_000_000
    ax.plot(df.index, history_mu, label="Generation data (observed and training imputations)", linewidth=1.2)
    ax.plot(test_index, forecast_kwh / 1_000_000, label="Prophet test forecast", linestyle="--", linewidth=2)
    ax.axvline(TEST_START, linestyle=":", linewidth=1.5, label="Test period starts")
    ax.set_title("Solar Generation History and Prophet Test Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy generated (MU)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "prophet_full_history_and_forecast.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: Prophet component chart. Components are on transformed log1p(kWh) scale.
    fig = model.plot_components(forecast_log)
    fig.suptitle("Prophet Components (Target Modelled on log1p(kWh) Scale)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "prophet_components_log_scale.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save fitted Prophet object in its supported JSON format.
    try:
        from prophet.serialize import model_to_json
        with open(outdir / "prophet_model.json", "w", encoding="utf-8") as f:
            f.write(model_to_json(model))
    except Exception as exc:
        with open(outdir / "model_serialization_note.txt", "w", encoding="utf-8") as f:
            f.write(f"Model JSON serialization was skipped: {exc}\n")

    config = {
        "input_file": input_path.name,
        "data_start": str(df.index.min().date()),
        "data_end": str(df.index.max().date()),
        "train_start": str(df.index[train_mask].min().date()),
        "train_end": str(df.index[train_mask].max().date()),
        "test_start": str(df.index[test_mask].min().date()),
        "test_end": str(df.index[test_mask].max().date()),
        "feature_set": args.feature_set,
        "feature_columns": feature_cols,
        "target_transform": "log1p(kWh)",
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "metrics": metrics,
    }
    (outdir / "prophet_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nPROPHET SOLAR GENERATION RESULTS")
    print("================================")
    print(f"Training: {df.index[train_mask].min().date()} to {df.index[train_mask].max().date()} ({int(train_mask.sum())} daily rows)")
    print(f"Test: {df.index[test_mask].min().date()} to {df.index[test_mask].max().date()} ({int(test_mask.sum())} daily rows)")
    print(f"Observed test targets scored: {metrics['n_scored_test_observations']}")
    print(f"Features/regressors: {len(feature_cols)}")
    print(f"MAE:  {metrics['MAE_kWh']:.3f} kWh  | {metrics['MAE_MU']:.6f} MU")
    print(f"RMSE: {metrics['RMSE_kWh']:.3f} kWh  | {metrics['RMSE_MU']:.6f} MU")
    print(f"MAPE: {metrics['MAPE_percent']:.4f}%")
    print(f"R2:   {metrics['R2']:.6f}")
    print(f"\nFiles saved in: {outdir}")
    print("Plots: prophet_actual_vs_predicted.png, prophet_error_plot.png, prophet_actual_vs_predicted_scatter.png, prophet_metrics_plot.png, prophet_full_history_and_forecast.png, prophet_components_log_scale.png")


if __name__ == "__main__":
    main()
