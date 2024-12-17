import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

def __int__(self, ticker, start, end):
    self.ticker = ticker
    self.start = start
    self.end = end
    
# Create sequences for training
def create_sequences(data, seq_length)-> np.array:
  sequences = []
  for i in range(len(data) - seq_length):
       sequences.append(data[i:i+seq_length])

  return np.array(sequences)

def Basic_LSTM(self)-> pd.DataFrame:
    """ 
    Build a basic LSTM Model
    """
    # Load and preprocess the data
    data = yf.download(self.ticker, start = self.start, end = self.end)
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
   
    # Prepare the training and test datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    
    seq_length = 60
    X_train = create_sequences(train_data, seq_length)
    X_test = create_sequences(test_data, seq_length)

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train the model
    model.fit(X_train, train_data[seq_length:], batch_size = 32, epochs = 5)

    # Predict stock prices
    X_test = create_sequences(test_data, seq_length)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    print(pd.DataFrame(predictions).describe())
