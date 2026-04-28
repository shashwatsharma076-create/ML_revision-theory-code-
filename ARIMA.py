import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("        TIME SERIES FORECASTING - ARIMA - 10 EXAMPLES")
print("="*70)

# ============================================================================
# EXAMPLE 1: Load Data & Explore
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Time Series Data & Exploration")
print("="*70)

np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=100, freq='M')

trend = np.linspace(10, 50, 100)
seasonal = 10 * np.sin(np.linspace(0, 20*np.pi, 100))
noise = np.random.normal(0, 2, 100)
data = trend + seasonal + noise

ts = pd.Series(data, index=dates)
print(f"Time series: {len(ts)} observations")
print(f"Date range: {ts.index[0]} to {ts.index[-1]}")
print(f"\nStatistics:")
print(ts.describe())

plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Sample Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('ts_original.png', dpi=100)
plt.close()
print("Saved: ts_original.png")

# ============================================================================
# EXAMPLE 2: Stationarity Test (ADF)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Augmented Dickey-Fuller Test")
print("="*70)

def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] < 0.05:
        print("→ Stationary (reject H0)")
    else:
        print("→ Non-stationary (fail to reject H0)")
    return result[1] < 0.05

print("\nTesting original series:")
is_stationary = adf_test(ts)

print("\nTesting first difference:")
ts_diff = ts.diff().dropna()
is_stationary_diff = adf_test(ts_diff)

# ============================================================================
# EXAMPLE 3: ACF & PACF Plots
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: ACF & PACF Analysis")
print("="*70)

acf_values = acf(ts_diff, nlags=20)
pacf_values = pacf(ts_diff, nlags=20)

print("ACF (first 10 lags):")
for i, val in enumerate(acf_values[:10]):
    print(f"   Lag {i}: {val:+.4f}")

print("\nPACF (first 10 lags):")
for i, val in enumerate(pacf_values[:10]):
    print(f"   Lag {i}: {val:+.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(ts_diff)
axes[0, 0].set_title('Differenced Series')

plot_acf(ts_diff, lags=20, ax=axes[0, 1])
axes[0, 1].set_title('ACF')

plot_pacf(ts_diff, lags=20, ax=axes[1, 0])
axes[1, 0].set_title('PACF')

axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=100)
plt.close()
print("\nSaved: acf_pacf.png")

# ============================================================================
# EXAMPLE 4: ARIMA Model Fitting
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: ARIMA Model Fitting")
print("="*70)

# Simple ARIMA(1,1,1)
model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()

print(fitted.summary().tables[0])
print(f"\nAIC: {fitted.aic:.4f}")
print(f"BIC: {fitted.bic:.4f}")

# ============================================================================
# EXAMPLE 5: Forecasting
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: ARIMA Forecasting")
print("="*70)

forecast = fitted.forecast(steps=12)

print("Next 12 forecasts:")
for i, (date, val) in enumerate(forecast.items()):
    print(f"   {date.strftime('%Y-%m')}: {val:.2f}")

# Prediction intervals
pred = fitted.get_prediction(start=len(ts), end=len(ts)+11)
conf = pred.conf_int(alpha=0.05)

print("\nForecasts with 95% CI:")
for i, val in enumerate(forecast):
    lower = conf.iloc[i, 0]
    upper = conf.iloc[i, 1]
    print(f"   {val:.2f} ({lower:.2f}, {upper:.2f})")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts, label='Historical')
plt.plot(forecast, label='Forecast', color='red')
plt.fill_between(forecast.index, conf.iloc[:, 0], conf.iloc[:, 1], 
               alpha=0.3, color='red')
plt.legend()
plt.title('ARIMA Forecast')
plt.tight_layout()
plt.savefig('arima_forecast.png', dpi=100)
plt.close()
print("\nSaved: arima_forecast.png")

# ============================================================================
# EXAMPLE 6: Model Selection (Grid Search)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Model Selection")
print("="*70)

results = []
for p in range(3):
    for d in range(2):
        for q in range(3):
            try:
                model = ARIMA(ts, order=(p, d, q))
                fitted = model.fit()
                results.append({
                    'order': (p, d, q),
                    'aic': fitted.aic,
                    'bic': fitted.bic
                })
            except:
                pass

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('aic')

print("Top 5 models by AIC:")
print(df_results.head())

best_order = df_results.iloc[0]['order']
print(f"\nBest order: {best_order}")

# ============================================================================
# EXAMPLE 7: Residual Analysis
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 7: Residual Analysis")
print("="*70)

residuals = fitted.resid

print(f"Residual statistics:")
print(f"   Mean: {residuals.mean():.4f}")
print(f"   Std: {residuals.std():.4f}")
print(f"   Skewness: {residuals.skew():.4f}")
print(f"   Kurtosis: {residuals.kurtosis():.4f}")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_fmt=True)
print(f"\nLjung-Box test (lag 10):")
print(f"   p-value: {lb_test[1][0]:.4f}")

if lb_test[1][0] > 0.05:
    print("   → No autocorrelation (good!)")

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals')
axes[0, 0].axhline(y=0, color='r', linestyle='--')

axes[0, 1].hist(residuals, bins=20, edgecolor='black')
axes[0, 1].set_title('Residual Histogram')

plot_acf(residuals, lags=20, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals')

axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('arima_residuals.png', dpi=100)
plt.close()
print("\nSaved: arima_residuals.png")

# ============================================================================
# EXAMPLE 8: Seasonal ARIMA (SARIMA)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 8: Seasonal ARIMA (SARIMA)")
print("="*70)

# Create seasonal data
np.random.seed(42)
n = 72
dates = pd.date_range('2015-01-01', periods=n, freq='M')
trend = np.linspace(10, 40, n)
seasonal = 15 * np.sin(np.linspace(0, 12*np.pi, n))
noise = np.random.normal(0, 2, n)
seasonal_data = trend + seasonal + noise

ts_seasonal = pd.Series(seasonal_data, index=dates)

print(f"Data: {len(ts_seasonal)} months (6 years)")

# SARIMA
model = SARIMAX(ts_seasonal, 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
fitted = model.fit()

print(f"\nSARIMA(1,1,1)(1,1,1)[12]")
print(f"AIC: {fitted.aic:.4f}")
print(f"BIC: {fitted.bic:.4f}")

# Forecast
forecast = fitted.forecast(steps=24)
print("\n24-month forecast:")
print(forecast.tail(6))

# ============================================================================
# EXAMPLE 9: Cross-Validation (Time Series Split)
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 9: Time Series Cross-Validation")
print("="*70)

train_size = 80
errors = []

for i in range(train_size, len(ts)-1):
    train = ts[:i]
    test = ts[i]
    
    model = ARIMA(train, order=(1, 1, 1))
    fitted = model.fit()
    
    pred = fitted.forecast(steps=1).iloc[0]
    error = test - pred
    errors.append(error)

errors = np.array(errors)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
mape = np.mean(np.abs(errors / ts[train_size:])) * 100

print(f"Time series CV (80 train → test):")
print(f"   MAE: {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAPE: {mape:.2f}%")

# ============================================================================
# EXAMPLE 10: Compare Models
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 10: Compare Forecasting Methods")
print("="*70)

# Split data
train = ts[:-12]
test = ts[-12:]

# Naive (last value)
naive_pred = np.full(12, train.iloc[-1])

# Moving average (last 3)
ma_window = 3
ma_pred = np.full(12, train.iloc[-ma_window:].mean())

# ARIMA
arima_model = ARIMA(train, order=(1, 1, 1))
arima_fitted = arima_model.fit()
arima_pred = arima_fitted.forecast(steps=12)

# Calculate errors
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Forecast Errors (12 steps):")
print(f"\nNaive:")
print(f"   MAE: {mean_absolute_error(test, naive_pred):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(test, naive_pred)):.4f}")

print(f"\nMoving Average (MA-3):")
print(f"   MAE: {mean_absolute_error(test, ma_pred):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(test, ma_pred)):.4f}")

print(f"\nARIMA(1,1,1):")
print(f"   MAE: {mean_absolute_error(test, arima_pred):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(test, arima_pred)):.4f}")

# Plot comparison
plt.figure(figsize=(12, 4))
plt.plot(train.index[-24:], train.iloc[-24:], label='Historical')
plt.plot(test.index, test, 'ko-', label='Actual')
plt.plot(test.index, naive_pred, 'r--', label='Naive')
plt.plot(test.index, ma_pred, 'g--', label='MA(3)')
plt.plot(test.index, arima_pred, 'b-', label='ARIMA')
plt.legend()
plt.title('Forecast Comparison')
plt.tight_layout()
plt.savefig('forecast_comparison.png', dpi=100)
plt.close()
print("\nSaved: forecast_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("                           SUMMARY")
print("="*70)
print("""
ARIMA Key Points:
────────────────
• ARIMA(p,d,q): Autoregressive + Integrated + Moving Average
• d = order of differencing (makes stationary)
• p = AR order (from PACF cutoff)
• q = MA order (from ACF cutoff)

Model Selection:
───────────────
• ADF test for stationarity
• ACF/PACF for initial (p,q)
• AIC/BIC for order selection

Forecasting:
─────────────
• Point forecasts for future values
• Prediction intervals widen with horizon
• Check residuals for model adequacy

Common Models:
──────────────
• ARIMA(1,1,1) - simplest
• ARIMA(2,1,2) - common choice
• SARIMA(p,d,q)(P,D,Q)[s] - seasonal data
""")

print("="*70)
print("                   EXAMPLES COMPLETE!")
print("="*70)