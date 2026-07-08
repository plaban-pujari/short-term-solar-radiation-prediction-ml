import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


print("Loading dataset...")

df = pd.read_csv("cleaned_data.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')
df.set_index('DateTime', inplace=True)




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


print("Training XGBoost...")

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 XGBOOST PERFORMANCE")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


joblib.dump(model, "xgb_model_final.pkl")
print("\n✅ Model saved as xgb_model_final.pkl")


plt.figure(figsize=(12,5))

plt.plot(df.index[split:], y_test, label="Actual")
plt.plot(df.index[split:], y_pred, label="Predicted")

plt.legend()
plt.title("XGBoost Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Solar Radiation")
plt.grid()

plt.show()


latest = X.iloc[-1].values.reshape(1,-1)

prediction = model.predict(latest)[0]

print(f"\n🌞 Next Step Prediction: {prediction:.2f} W/m²")