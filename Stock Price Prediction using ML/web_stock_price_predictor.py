import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# Input for stock ticker
stock = st.text_input("Enter a stock ID", "GOOG")

# Date range for historical data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download historical data
google_data = yf.download(stock, start, end)

if not google_data.empty:
    st.subheader("Stock Data")
    st.write(google_data)

    # Load the model
    model = load_model("Latest_stock_model.keras")

    # Splitting the data
    splitting_len = int(len(google_data) * 0.7)
    train_data = google_data[:splitting_len]
    test_data = google_data[splitting_len:]

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(google_data['Close'].values.reshape(-1, 1))

    # Create test dataset
    test_inputs = scaled_data[splitting_len - 60:]  # 60 timesteps
    x_test = []
    for i in range(60, len(test_inputs)):
        x_test.append(test_inputs[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Function to plot the graph
    def plot_graph (figsize, predicted_values, actual_values, extra_data = 0, extra_dataset= None):
        fig = plt.figure(figsize=figsize)
        plt.plot(actual_values, 'b', label='Actual')
        plt.plot(predicted_values, 'orange', label='Predicted')
        plt.legend()
        if extra_data:
            plt.plot(extra_dataset)
        return fig

    # Plotting the actual and predicted values
    actual_values = test_data['Close'].values
    fig = plot_graph((10, 5), predictions, actual_values)
    st.pyplot(fig)



#to run project use this 2 command
#cd "c:\Users\amrap\OneDrive\Desktop\Stock Price Prediction using ML"
#streamlit run "web_stock_price_predictor.py"


st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days']=google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_250_days'],google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days']=google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_200_days'],google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days']=google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])
print(type(x_test))
print(x_test.columns)

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    x_data.append(scaled_data[i])

x_data, y_data = np.array(x_data),np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)

inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
{
'original_test_data':inv_y_test.reshape(-1),
   'predictions': inv_pre.reshape(-1)
},
    index= google_data.index[splitting_len+100:]
)

st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader("Original close price vs Close price")
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data- not used", "Original test data", "Predicted Test data"])
st.pyplot(fig)
