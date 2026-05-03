import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("        PROPHET FORECASTING - 10 EXAMPLES")
print("="*70)

# ============================================================================
# EXAMPLE 1: Basic Prophet Forecast
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 1: Basic Prophet Forecast")
print("="*70)

np.random.seed(42)
n = 500
dates = pd.date_range('2022-01-01', periods=n, freq='D')

trend = np.linspace(100, 150, n)
seasonal = 20 * np.sin(np.linspace(0, 20*np.pi, n))
noise = np.random.normal(0, 5, n)
y = trend + seasonal + noise

df = pd.DataFrame({'ds': dates, 'y': y})

print(f"Data: {len(df)} daily observations")
print(f"Date range: {df.ds.min()} to {df.ds.max()}")

# Fit Prophet
m = Prophet()
m.fit(df)

print("✓ Model fitted")

# ============================================================================
# EXAMPLE 2: Future Forecast
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Forecast Future Values")
print("="*70)

future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

print(f"Forecast: {len(forecast)} rows (historical + 30 future)")
print("\nFuture predictions:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

# Plot
fig1 = m.plot(forecast)
fig1.suptitle('Prophet Forecast', fontsize=12)
fig1.savefig('prophet_forecast.png', dpi=100)
plt.close(fig1)
print("\nSaved: prophet_forecast.png")

# ============================================================================
# EXAMPLE 3: Decompose Components
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Component Decomposition")
print("="*70)

fig2 = m.plot_components(forecast)
fig2.savefig('prophet_components.png', dpi=100)
plt.close(fig2)
print("Saved: prophet_components.png")

print("\nComponents in forecast:")
print(f"  Trend (last value): {forecast.yhat.iloc[-30:].mean():.2f}")
print(f"  Weekly (max-min): {forecast.weekly.max() - forecast.weekly.min():.2f}")
print(f"  Yearly (max-min): {forecast.yearly.max() - forecast.yearly.min():.2f}")

# ============================================================================
# EXAMPLE 4: Custom Seasonality
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: Custom Seasonality")
print("="*70)

m_custom = Prophet(yearly_seasonality=False, weekly_seasonality=False)
m_custom.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m_custom.add_seasonality(name='weekly', period=7, fourier_order=3)
m_custom.fit(df)

future_custom = m_custom.make_future_dataframe(periods=30)
forecast_custom = m_custom.predict(future_custom)

print("Custom seasonalities added:")
print("  - Monthly (30.5 days, 5 Fourier terms)")
print("  - Weekly (7 days, 3 Fourier terms)")

fig3 = m_custom.plot_components(forecast_custom)
fig3.savefig('prophet_custom_seasonality.png', dpi=100)
plt.close(fig3)
print("\nSaved: prophet_custom_seasonality.png")

# ============================================================================
# EXAMPLE 5: Holiday Effects
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: Holiday Effects")
print("="*70)

# Custom holidays
holidays = pd.DataFrame({
    'holiday': ['Black Friday', 'Christmas', 'New Year'],
    'ds': pd.to_datetime([
        '2022-11-25', '2022-12-25', '2023-01-01',
        '2023-11-24', '2023-12-25', '2024-01-01',
        '2024-11-29', '2024-12-25', '2025-01-01'
    ]),
    'lower_window': -2,
    'upper_window': 1
})

m_holidays = Prophet(holidays=holidays, holidays_prior_scale=15)
m_holidays.fit(df)

future_h = m_holidays.make_future_dataframe(periods=30)
forecast_h = m_holidays.predict(future_h)

print("Holiday effects:")
print(forecast_h[['ds', 'holidays']].tail(15).dropna())

fig4 = m_holidays.plot_components(forecast_h)
fig4.savefig('prophet_holidays.png', dpi=100)
plt.close(fig4)
print("\nSaved: prophet_holidays.png")

# ============================================================================
# EXAMPLE 6: Changepoints
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Changepoint Analysis")
print("="*70)

# Get changepoints
changepoints = m.changepoints
print(f"Number of changepoints: {len(changepoints)}")
print(f"Changepoint range: {m.changepoint_range}")

# Trend plot with changepoints
plt.figure(figsize=(10, 4))
plt.plot(df.ds, df.y, 'k.', alpha=0.3, label='Data')
plt.plot(forecast.ds, forecast.trend, 'b-', linewidth=2, label='Trend')

for cp in changepoints:
    plt.axvline(cp, color='r', alpha=0.3)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Trend with Changepoints (red lines)')
plt.legend()
plt.tight_layout()
plt.savefig('prophet_changepoints.png', dpi=100)
plt.close()
print("Saved: prophet_changepoints.png")

# ============================================================================
# EXAMPLE 7: Hyperparameter Tuning
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 7: Hyperparameter Tuning")
print("="*70)

configs = [
    {'changepoint_prior_scale': 0.01},
    {'changepoint_prior_scale': 0.05},
    {'changepoint_prior_scale': 0.1},
]

train = df[:-30]
test = df[-30:]

for cfg in configs:
    m_test = Prophet(**cfg)
    m_test.fit(train)
    
    future_t = m_test.make_future_dataframe(periods=30)
    forecast_t = m_test.predict(future_t)
    
    preds = forecast_t.yhat[-30:].values
    actual = test.y.values
    rmse = np.sqrt(np.mean((preds - actual)**2))
    
    print(f"changepoint_prior_scale={cfg['changepoint_prior_scale']}: RMSE={rmse:.2f}")

# ============================================================================
# EXAMPLE 8: Multiplicative Seasonality
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 8: Multiplicative Seasonality")
print("="*70)

# Create data with multiplicative seasonality
np.random.seed(42)
trend_mult = np.linspace(50, 200, n)
seasonal_mult = trend_mult * 0.2 * np.sin(np.linspace(0, 20*np.pi, n))
noise_mult = np.random.normal(0, 5, n)
y_mult = trend_mult + seasonal_mult + noise_mult

df_mult = pd.DataFrame({'ds': dates, 'y': y_mult})

m_mult = Prophet(seasonality_mode='multiplicative')
m_mult.fit(df_mult)

future_mult = m_mult.make_future_dataframe(periods=30)
forecast_mult = m_mult.predict(future_mult)

print("Multiplicative seasonality fitted")
print(f"Seasonal amplitude grows with trend")

fig5 = m_mult.plot_components(forecast_mult)
fig5.savefig('prophet_multiplicative.png', dpi=100)
plt.close(fig5)
print("\nSaved: prophet_multiplicative.png")

# ============================================================================
# EXAMPLE 9: Uncertainty Intervals
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 9: Uncertainty Intervals")
print("="*70)

m_80 = Prophet(interval_width=0.8)
m_80.fit(df)

m_95 = Prophet(interval_width=0.95)
m_95.fit(df)

future_u = m_80.make_future_dataframe(periods=30)
forecast_80 = m_80.predict(future_u)
forecast_95 = m_95.predict(future_u)

print("80% confidence interval (last 5 days):")
print(forecast_80[['ds', 'yhat_lower', 'yhat_upper']].tail(5))

print("\n95% confidence interval (last 5 days):")
print(forecast_95[['ds', 'yhat_lower', 'yhat_upper']].tail(5))

# Plot comparison
plt.figure(figsize=(10, 4))
plt.plot(df.ds[-60:], df.y[-60:], 'k.', alpha=0.5, label='Data')
plt.plot(forecast_80.ds[-30:], forecast_80.yhat[-30:], 'b-', label='Forecast')
plt.fill_between(forecast_80.ds[-30:], forecast_80.yhat_lower[-30:],
                forecast_80.yhat_upper[-30:], alpha=0.3, color='b', label='80% CI')
plt.fill_between(forecast_95.ds[-30:], forecast_95.yhat_lower[-30:],
                forecast_95.yhat_upper[-30:], alpha=0.2, color='green', label='95% CI')
plt.legend()
plt.title('Forecast with Uncertainty Intervals')
plt.tight_layout()
plt.savefig('prophet_uncertainty.png', dpi=100)
plt.close()
print("\nSaved: prophet_uncertainty.png")

# ============================================================================
# EXAMPLE 10: Cross-Validation
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 10: Cross-Validation")
print("="*70)

m_cv = Prophet()
m_cv.fit(df)

df_cv = cross_validation(m_cv,
                        initial='365 days',
                        period='90 days',
                        horizon='30 days')

df_p = performance_metrics(df_cv)

print(f"Cross-validation folds: {df_cv.cutoff.nunique()}")
print(f"Horizon: 30 days")
print(f"\nPerformance metrics:")
print(f"  MAE: {df_p.mae.mean():.2f}")
print(f"  RMSE: {df_p.rmse.mean():.2f}")
print(f"  MAPE: {df_p.mape.mean():.4f}")

# Plot performance
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df_p.horizon.dt.days, df_p.mae, 'bo')
plt.xlabel('Horizon (days)')
plt.ylabel('MAE')
plt.title('MAE vs Horizon')

plt.subplot(1, 2, 2)
plt.plot(df_p.horizon.dt.days, df_p.rmse, 'ro')
plt.xlabel('Horizon (days)')
plt.ylabel('RMSE')
plt.title('RMSE vs Horizon')

plt.tight_layout()
plt.savefig('prophet_cv.png', dpi=100)
plt.close()
print("\nSaved: prophet_cv.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("                           SUMMARY")
print("="*70)
print("""
Prophet Key Points:
─────────────────
• Additive: y(t) = g(t) + s(t) + h(t) + εₜ
• Multiplicative: y(t) = g(t) × [1 + s(t) + h(t)] + εₜ

Model Components:
────────────────
• Trend (linear or logistic)
• Seasonality (weekly, yearly, custom)
• Holidays (built-in or custom)

Key Parameters:
──────────────
• changepoint_prior_scale (default: 0.05)
• seasonality_prior_scale (default: 10.0)
• holidays_prior_scale (default: 10.0)
• interval_width (default: 0.8)

Workflow:
─────────
1. Prepare data (ds, y columns)
2. Fit Prophet model
3. Make future dataframe
4. Predict
5. Check diagnostics
6. Cross-validate
""")

print("="*70)
print("                   EXAMPLES COMPLETE!")
print("="*70)