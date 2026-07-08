import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_data.csv")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime')
df.set_index('DateTime', inplace=True)

df = df[['radiation', 'temperature', 'humidity', 'wind_speed']]


df['temperature'] = df['temperature'].replace(-40, np.nan)
df = df[(df['radiation'] >= 0) & (df['radiation'] <= 1200)]
df.ffill(inplace=True)




for lag in [1, 2, 3]:
    df[f'lag{lag}'] = df['radiation'].shift(lag)


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


from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))


def create_sequences(X, y, time_steps=6):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 6

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)


import tensorflow as tf

lstm_model = tf.keras.Sequential([
    tf.keras.Input(shape=(time_steps, X_train.shape[1])),

    tf.keras.layers.LSTM(80, return_sequences=True),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.LSTM(40),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1)
])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
    loss='mse'
)


early_stop = tf.keras.callbacks.EarlyStopping(
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=3,
    min_lr=1e-5
)


lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=60,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


y_train_pred_scaled = lstm_model.predict(X_train_seq)
y_test_pred_scaled = lstm_model.predict(X_test_seq)

y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

y_train_actual = y_train.values[time_steps:]
y_test_actual = y_test.values[time_steps:]


residual_train = y_train_actual - y_train_pred
residual_test = y_test_actual - y_test_pred


residual_train = np.clip(residual_train, -150, 150)


from xgboost import XGBRegressor

X_train_res = pd.DataFrame(X_train_scaled[time_steps:], columns=features)
X_test_res = pd.DataFrame(X_test_scaled[time_steps:], columns=features)

X_train_res['lstm_pred'] = y_train_pred
X_test_res['lstm_pred'] = y_test_pred

xgb_model = XGBRegressor(
    n_estimators=220,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=4,
    reg_alpha=1,
    random_state=42
)

xgb_model.fit(X_train_res, residual_train)


residual_pred = xgb_model.predict(X_test_res)
y_final = y_test_pred + residual_pred


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("\n===== LSTM PERFORMANCE =====")
print("R2:", r2_score(y_test_actual, y_test_pred))
print("MAE:", mean_absolute_error(y_test_actual, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_test_pred)))

print("\n===== HYBRID PERFORMANCE =====")
print("R2:", r2_score(y_test_actual, y_final))
print("MAE:", mean_absolute_error(y_test_actual, y_final))
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_final)))


plt.figure(figsize=(12,5))

plt.plot(df.index[split+time_steps:], y_test_actual, label="Actual")
plt.plot(df.index[split+time_steps:], y_test_pred, label="LSTM")
plt.plot(df.index[split+time_steps:], y_final, label="Hybrid")

plt.legend()
plt.title("Tuned Hybrid Model (No Feature Change)")
plt.grid()
plt.show()


import joblib

lstm_model.save("lstm_model.keras")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("\n✅ Tuned Hybrid Model Saved Successfully!")