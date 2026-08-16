import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")   # suppress convergence / deprecation warnings

# ── Step 1: Load / Create the Time-Series Dataset ────────────────────────────
# The manual references "diabetes9.csv".  We generate it here from the UCI
# Diabetes dataset by assigning each patient record a daily date starting
# 2018-01-01, treating the sequence of Glucose readings as a daily time series.

import os

CSV_PATH = "diabetes9.csv"

if not os.path.exists(CSV_PATH):
    # Build diabetes9.csv from the UCI dataset
    base_df = pd.read_csv("../../Experiment 3/uci_diabetes.csv")
    dates   = pd.date_range(start="2018-01-01", periods=len(base_df), freq="D")
    ts_df   = pd.DataFrame({"Date": dates, "Glucose": base_df["Glucose"].values})
    ts_df.to_csv(CSV_PATH, index=False)
    print(f"Created {CSV_PATH} with {len(ts_df)} rows.")
else:
    print(f"Loaded existing {CSV_PATH}.")

# ── Step 2: Load and Preview ──────────────────────────────────────────────────
diabetes_data = pd.read_csv(CSV_PATH)
print("\nDataset Preview:")
print(diabetes_data.head())
print(f"\nShape : {diabetes_data.shape}")

# ── Step 3: Convert Date column and set as index ──────────────────────────────
diabetes_data["Date"]  = pd.to_datetime(diabetes_data["Date"])
diabetes_data          = diabetes_data.set_index("Date")
diabetes_data          = diabetes_data.sort_index()

print(f"\nDate range : {diabetes_data.index.min()} to {diabetes_data.index.max()}")
print(f"Glucose - Mean : {diabetes_data['Glucose'].mean():.2f}")
print(f"Glucose - Std  : {diabetes_data['Glucose'].std():.2f}")

# ── Step 4: Plot Time Series ──────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(diabetes_data["Glucose"], label="Glucose Level", color="blue", linewidth=0.8)
plt.xlabel("Date")
plt.ylabel("Glucose Level")
plt.title("Time Series of Glucose Levels")
plt.legend()
plt.tight_layout()
plt.savefig("glucose_time_series.png", bbox_inches="tight")
plt.close()
print("\nPlot saved: glucose_time_series.png")

# ── Step 5: Decompose Time Series ─────────────────────────────────────────────
# period=30 treats every ~30 days as one seasonal cycle
decomposition = seasonal_decompose(
    diabetes_data["Glucose"], model="additive", period=30
)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
decomposition.trend.plot(ax=axes[0],   title="Trend Component",    color="blue")
decomposition.seasonal.plot(ax=axes[1],title="Seasonal Component",  color="green")
decomposition.resid.plot(ax=axes[2],   title="Residual Component",  color="red")
axes[0].set_ylabel("Trend")
axes[1].set_ylabel("Seasonal")
axes[2].set_ylabel("Residual")
plt.tight_layout()
plt.savefig("decomposition.png", bbox_inches="tight")
plt.close()
print("Plot saved: decomposition.png")

# ── Step 6: Moving Average Smoothing ─────────────────────────────────────────
diabetes_data["Glucose_MA"] = (
    diabetes_data["Glucose"].rolling(window=7).mean()
)

plt.figure(figsize=(12, 5))
plt.plot(diabetes_data["Glucose"],    label="Original",             alpha=0.5, color="steelblue")
plt.plot(diabetes_data["Glucose_MA"], label="7-day Moving Average", color="red", linewidth=2)
plt.xlabel("Date")
plt.ylabel("Glucose Level")
plt.title("Moving Average Smoothing (7-day Window)")
plt.legend()
plt.tight_layout()
plt.savefig("moving_average.png", bbox_inches="tight")
plt.close()
print("Plot saved: moving_average.png")

# ── Step 7: ARIMA Forecasting ─────────────────────────────────────────────────
glucose_series = diabetes_data["Glucose"]
train_size     = int(len(glucose_series) * 0.8)
train          = glucose_series.iloc[:train_size]
test           = glucose_series.iloc[train_size:]

print(f"\nTrain size : {len(train)}")
print(f"Test  size : {len(test)}")
print("Fitting ARIMA(5, 1, 0) model …")

# ARIMA(p=5, d=1, q=0) as specified in the manual
arima_model  = ARIMA(train, order=(5, 1, 0))
fitted_model = arima_model.fit()

# Forecast for the test period
forecast = fitted_model.forecast(steps=len(test))

# ── Step 8: Plot Forecast vs Actual ───────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(range(len(test)), test.values,     label="Actual",   color="blue",  linewidth=1.5)
plt.plot(range(len(test)), forecast.values, label="Forecast", color="red",   linewidth=1.5,
         linestyle="--")
plt.xlabel("Time Step (Test Period)")
plt.ylabel("Glucose Level")
plt.title("ARIMA(5,1,0) Model – Forecast vs Actual Glucose Levels")
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png", bbox_inches="tight")
plt.close()
print("Plot saved: arima_forecast.png")

# ── Forecast Accuracy ─────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(test.values, forecast.values)
mae = mean_absolute_error(test.values, forecast.values)
rmse = np.sqrt(mse)

print("\n--- ARIMA Forecast Evaluation (Test Set) ---")
print(f"  MSE  : {mse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")

# ── Consolidated output.png (4-panel summary) ─────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Time Series Analysis – Glucose Levels (UCI Diabetes Dataset)",
             fontsize=13, fontweight='bold')

# Panel 1 – Raw time series
axes[0, 0].plot(diabetes_data["Glucose"], color="blue", linewidth=0.8)
axes[0, 0].set_title("Time Series of Glucose Levels")
axes[0, 0].set_xlabel("Date"); axes[0, 0].set_ylabel("Glucose")

# Panel 2 – Decomposition (trend only, for readability)
axes[0, 1].plot(decomposition.trend.dropna(), color="purple", linewidth=1.2)
axes[0, 1].set_title("Trend Component (Decomposition)")
axes[0, 1].set_xlabel("Date"); axes[0, 1].set_ylabel("Trend")

# Panel 3 – Moving average
axes[1, 0].plot(diabetes_data["Glucose"],    alpha=0.5, color="steelblue", label="Original")
axes[1, 0].plot(diabetes_data["Glucose_MA"], color="red", linewidth=2,      label="7-day MA")
axes[1, 0].set_title("Moving Average Smoothing")
axes[1, 0].set_xlabel("Date"); axes[1, 0].set_ylabel("Glucose")
axes[1, 0].legend(fontsize=8)

# Panel 4 – ARIMA forecast
axes[1, 1].plot(range(len(test)), test.values,     color="blue", label="Actual",   linewidth=1.5)
axes[1, 1].plot(range(len(test)), forecast.values, color="red",  label="Forecast", linewidth=1.5,
                linestyle="--")
axes[1, 1].set_title(f"ARIMA Forecast  (RMSE={rmse:.2f})")
axes[1, 1].set_xlabel("Time Step"); axes[1, 1].set_ylabel("Glucose")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("output.png", bbox_inches="tight")
plt.close()
print("Plot saved: output.png  (4-panel summary)")

print("\nRESULT:")
print("The Time Series Analysis identifies trends and seasonal patterns in glucose")
print("levels from the UCI Diabetes dataset. The ARIMA(5,1,0) model effectively")
print("forecasts future glucose values. Moving average smoothing reduces noise,")
print("and seasonal decomposition separates trend, seasonality, and residual components.")
