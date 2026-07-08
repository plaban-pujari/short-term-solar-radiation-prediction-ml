import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("cleaned_data.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')
df.set_index('DateTime', inplace=True)


df = df[
    (df['radiation'] >= 0) & (df['radiation'] <= 1200)
]



df['lag1'] = df['radiation'].shift(1)
df['lag2'] = df['radiation'].shift(2)
df['lag3'] = df['radiation'].shift(3)


df['rolling_mean_3'] = df['radiation'].shift(1).rolling(3).mean()


df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['day_of_year'] = df.index.dayofyear


df['target'] = df['radiation'].shift(-1)

df.dropna(inplace=True)


features = [
    'lag1','lag2','lag3',
    'rolling_mean_3',
    'hour_sin','hour_cos','day_of_year',
    'temperature','humidity','wind_speed'
]

X = df[features]
y = df['target']


split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 TEST PERFORMANCE")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


y_train_pred = model.predict(X_train)

r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n📊 TRAIN PERFORMANCE")
print(f"R² Train : {r2_train:.4f}")
print(f"RMSE Train: {rmse_train:.2f}")


joblib.dump(model, "rf_model_final.pkl")
print("\n✅ Model saved as rf_model_final.pkl")


plt.figure(figsize=(12,6))

plt.plot(df.index[split:], y_test, label="Actual", linewidth=2)
plt.plot(df.index[split:], y_pred, label="Predicted", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Solar Radiation")
plt.title("Random Forest Prediction vs Actual")
plt.legend()
plt.grid(True)

plt.show()