import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.express as px
from pyngrok import ngrok

amazon_stock_price_df = pd.read_csv('/content/mydata.csv')

st.set_page_config(page_title = 'Time Series Dashboard',
                   page_icon =':date:',
                   layout = 'wide')
st.title('Dashboard for Stock Price Prediction 	:chart:')

#moving page title upwards and in center
st.markdown(
    '''<style>
            div.block-container{
              padding-top:2rem;
              text-align:center;
            }
       </style>
    ''', 
    unsafe_allow_html = True)
#including important columns
amazon_stock_price_df = amazon_stock_price_df[['Date', 'Open', 'High', 'Low', 'Close']]
#converting date column to 'datetime' datatype
amazon_stock_price_df['Date'] = pd.to_datetime(amazon_stock_price_df['Date'])
#making the 'date' column as index
amazon_stock_price_df.set_index('Date', inplace = True)

import pandas as pd
col1, col2 = st.columns((2))

#getting min and max date
startDate = pd.to_datetime(amazon_stock_price_df.index).min()
endDate = pd.to_datetime(amazon_stock_price_df.index).max()

with col1:
  date1 = pd.to_datetime(st.date_input("Start Date", startDate))
with col2:
  date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = amazon_stock_price_df.loc[date1:date2]
#data normalization
scaler = MinMaxScaler()
data = scaler.fit_transform(df[df.columns])
WINDOW_SIZE = 20
def create_sliding_window(data, window_size):
  X, y = [], []
  for i in range(len(data) - window_size):
    X.append(data[i:i + window_size, :])  # Use all features
    y.append(data[i + window_size, :])    # Predict all columns (Open, High, Low, Close)
  return np.array(X), np.array(y)
X,  y = create_sliding_window(data, WINDOW_SIZE)
model = tf.keras.Sequential([
    # CONV 1D LAYER
    tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation = 'relu', input_shape = (X.shape[1], X.shape[2])),
    # FLATTENING LAYER
    tf.keras.layers.Flatten(),
    # DENSE HIDDEN LAYER
    tf.keras.layers.Dense(50, activation = 'relu'),
    # DENSE OUTPUT LAYER
    tf.keras.layers.Dense(X.shape[2])
])
# COMPILATION
model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['accuracy'])
#TRAINING THE MODEL
model.fit(X, y, epochs = 25, batch_size = 8)
preds = model.predict(X)

preds = scaler.inverse_transform(preds)

# Plot actual vs predicted prices using Plotly
for idx, col in enumerate(amazon_stock_price_df.columns):
    st.subheader(f'{col}')
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col],
        mode='lines',
        name=f'Actual {col}'
    ))

    # Plot predicted data
    fig.add_trace(go.Scatter(
        x=df.index[:len(preds)], 
        y=preds[:, idx],
        mode='lines',
        name=f'Predicted {col}',
        line=dict(color = 'yellow')
    ))

    st.plotly_chart(fig)
