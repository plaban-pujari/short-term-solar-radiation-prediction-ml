import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle


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


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))


split = int(len(X_scaled) * 0.8)

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)


early_stop = EarlyStopping(
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


y_pred_scaled = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)


r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n📊 ANN PERFORMANCE")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


model.save("ann_model.keras")

with open("scaler_X_ann.pkl", "wb") as f:
    pickle.dump(scaler_X, f)

with open("scaler_y_ann.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

print("\n✅ ANN Model Saved Successfully!")


plt.figure(figsize=(12,5))
plt.plot(df.index[split:], y_true, label="Actual")
plt.plot(df.index[split:], y_pred, label="Predicted")

plt.legend()
plt.title("ANN Prediction vs Actual")
plt.xlabel("Time")
plt.ylabel("Solar Radiation")
plt.grid()
plt.show()