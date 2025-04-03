import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("bitcoin_prices_2025.csv")

# Convert 'open_time' and 'close_time' to datetime objects
df['open_time'] = pd.to_datetime(df['open_time'])
df['close_time'] = pd.to_datetime(df['close_time'])

# Preserve original 'close' prices for plotting
df['close_original'] = df['close']

# Handle missing values (if any)
df.ffill(inplace=True)

# Feature Scaling
scaler_all = MinMaxScaler()  # For all features except 'close'
close_scaler = MinMaxScaler()  # Separate scaler for 'close'
features_to_scale = ['open', 'high', 'low', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']
df[features_to_scale] = scaler_all.fit_transform(df[features_to_scale])
df[['close']] = close_scaler.fit_transform(df[['close']])  # Scale 'close' separately

# Display the first few rows of the preprocessed data
print(df.head())

# Save the preprocessed data to a new CSV file
df.to_csv("bitcoin_prices_preprocessed.csv", index=False)

# Feature Engineering
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['Lag_1'] = df['close'].shift(1)
df['Lag_2'] = df['close'].shift(2)

# Drop rows with NaN values resulting from feature engineering
df.dropna(inplace=True)

# Plotting historical data with moving averages
plt.figure(figsize=(14, 7))
plt.plot(df['open_time'], df['close_original'], label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Historical Price')
plt.legend()
plt.show()

# Display the first few rows of the data with new features
print(df.head())

# Prepare data for LSTM
X = df[['close', 'SMA_20', 'SMA_50', 'Lag_1', 'Lag_2']].values
y = df['close'].values

# Reshape data for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Build LSTM model with Input layer
model = Sequential()
model.add(Input(shape=(1, 5)))  # time_steps=1, features=5
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model on the entire dataset
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Function to generate future dates
def generate_future_dates(start_date, periods):
    date_list = [start_date + timedelta(days=x) for x in range(1, periods + 1)]
    return date_list

# Make predictions for the next 4 months (approx. 120 days)
future_periods = 120
last_date = df['open_time'].iloc[-1]
future_dates = generate_future_dates(last_date, future_periods)

# Initialize future predictions
future_predictions = []
temp_df = df.tail(50).copy()  # Keep last 50 rows for SMA_50 calculation
last_known_close = temp_df['close'].iloc[-1]

for i in range(future_periods):
    # Calculate features using the last 50 rows
    sma_20 = temp_df['close'].tail(20).mean()
    sma_50 = temp_df['close'].tail(50).mean()
    lag_1 = temp_df['close'].iloc[-1]
    lag_2 = temp_df['close'].iloc[-2] if len(temp_df) >= 2 else lag_1

    # Prepare input data for prediction
    input_data = np.array([[last_known_close, sma_20, sma_50, lag_1, lag_2]])
    input_data = input_data.reshape((1, 1, 5))

    # Make prediction
    prediction = model.predict(input_data, verbose=0)[0][0]
    future_predictions.append(prediction)

    # Append the predicted close to temp_df for future SMA calculations
    new_row = pd.DataFrame({
        'open_time': [future_dates[i]],
        'close': [prediction],
        'SMA_20': [np.nan],
        'SMA_50': [np.nan],
        'Lag_1': [last_known_close],
        'Lag_2': [lag_1]
    })
    temp_df = pd.concat([temp_df, new_row], ignore_index=True)
    temp_df['SMA_20'] = temp_df['close'].rolling(window=20).mean()
    temp_df['SMA_50'] = temp_df['close'].rolling(window=50).mean()
    temp_df.ffill(inplace=True)  # Forward fill for any remaining NaNs

    # Update last_known_close for the next iteration
    last_known_close = prediction

# Inverse transform predictions to original scale
future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plotting historical and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(df['open_time'], df['close_original'], label='Historical Price', color='blue')
plt.plot(future_dates, future_predictions, label='Predicted Price', color='orange', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Price Prediction for the Next 4 Months')
plt.legend()
plt.show()