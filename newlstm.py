import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


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

X_all = df[features].values
y_all = df['target'].values


SEQ_LEN = 8   

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)


split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


ns, ts, nf = X_train.shape

X_train_2d = X_train.reshape(ns*ts, nf)
X_test_2d = X_test.reshape(X_test.shape[0]*ts, nf)

scaler_X = MinMaxScaler()
scaler_X.fit(X_train_2d)

X_train_scaled = scaler_X.transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))


model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(SEQ_LEN, nf)),
    Dropout(0.2),

    LSTM(16),
    Dropout(0.1),

    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)


es = EarlyStopping(patience=10, restore_best_weights=True)
rlr = ReduceLROnPlateau(factor=0.5, patience=5)

model.fit(
    X_train_scaled, y_train_scaled,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)


y_pred_scaled = model.predict(X_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true = y_test


r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n📊 LSTM PERFORMANCE")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")


dates = df.index[-len(y_true):]

plt.figure(figsize=(12,5))

plt.plot(dates, y_true, label="Actual", linewidth=2)
plt.plot(dates, y_pred, label="Predicted", linewidth=2)

plt.title("LSTM: Actual vs Predicted Solar Radiation")
plt.xlabel("Time")
plt.ylabel("Radiation (W/m²)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("LSTM_Actual_vs_Predicted.png", dpi=200)
plt.show()


model.save("lstm_model.keras")
joblib.dump(scaler_X, "scaler_X_lstm.pkl")
joblib.dump(scaler_y, "scaler_y_lstm.pkl")

print("\n✅ LSTM Model Saved Successfully!")