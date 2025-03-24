import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv("bitcoin_prices_2025.csv")

# Convert 'open_time' and 'close_time' to datetime objects
df['open_time'] = pd.to_datetime(df['open_time'])
df['close_time'] = pd.to_datetime(df['close_time'])

# Handle missing values (if any)
df.fillna(method='ffill', inplace=True)

# Feature Scaling
scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume']])

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

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 7))
plt.plot(df['open_time'], df['close'], label='Closing Price')
plt.plot(df['open_time'], df['SMA_20'], label='20-day SMA')
plt.plot(df['open_time'], df['SMA_50'], label='50-day SMA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Bitcoin Price with Moving Averages')
plt.legend()
plt.show()

# Display the first few rows of the data with new features
print(df.head())

# Prepare data for LSTM
X = df[['close', 'SMA_20', 'SMA_50', 'Lag_1', 'Lag_2']].values
y = df['close'].values

# Reshape data for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, X.shape[1])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model on the entire dataset
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Function to generate future dates
def generate_future_dates(start_date, periods):
    date_list = [start_date + timedelta(days=x) for x in range(periods)]
    return date_list

# Make predictions for the next 4 months (approx. 120 days)
future_periods = 120
last_date = df['open_time'].iloc[-1]
future_dates = generate_future_dates(last_date, future_periods)

# Create dummy data for future predictions
future_predictions = []
last_known_close = df['close'].iloc[-1]

for _ in range(future_periods):
    # Create a temporary DataFrame for the future data point
    temp_df = pd.DataFrame({
        'close': [last_known_close],
        'SMA_20': [np.nan],
        'SMA_50': [np.nan],
        'Lag_1': [np.nan],
        'Lag_2': [np.nan]
    })

    # Calculate SMA and lagged values
    temp_df['SMA_20'] = temp_df['close'].rolling(window=20).mean()
    temp_df['SMA_50'] = temp_df['close'].rolling(window=50).mean()
    temp_df['Lag_1'] = last_known_close
    temp_df['Lag_2'] = last_known_close  # Using last_known_close for simplicity

    # Fill NaN values
    temp_df.fillna(method='bfill', inplace=True)

    # Prepare the input for prediction
    future_data = temp_df[['close', 'SMA_20', 'SMA_50', 'Lag_1', 'Lag_2']].values
    future_data = future_data.reshape((1, 1, 5))

    # Make prediction
    prediction = model.predict(future_data, verbose=0)[0][0]
    future_predictions.append(prediction)

    # Update last_known_close with the predicted value
    last_known_close = prediction

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df['open_time'], df['close'], label='Historical Price')
plt.plot(future_dates, future_predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction for the Next 4 Months')
plt.legend()
plt.show()
