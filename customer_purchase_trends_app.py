"""
Customer Purchases Trend & Prediction App (Streamlit)

Features:
- Upload customer transactions CSV or use sample data
- Aggregate purchase trends by day
- Visualize total purchases and number of transactions
- Forecast future purchases using Linear Regression
- Download CSV reports of trends and forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='Customer Purchase Trends & Prediction', layout='wide')
st.title('📈 Customer Purchase Trends & Prediction Dashboard')

# ---------------- Sidebar ----------------
st.sidebar.header('Data Options')
use_sample = st.sidebar.checkbox('Use sample transaction data', value=True)
uploaded_file = st.sidebar.file_uploader('Upload transactions CSV', type=['csv'])

# ---------------- Sample Data ----------------
def sample_transaction_data():
    dates = pd.date_range('2025-01-01', periods=30)
    data = []
    for date in dates:
        for i in range(np.random.randint(5,15)):
            customer_id = np.random.randint(1,21)
            purchase_value = np.random.randint(50,500)
            data.append([customer_id, date, purchase_value])
    return pd.DataFrame(data, columns=['customer_id','date','purchase_value'])

# ---------------- Load Data ----------------
if use_sample:
    df = sample_transaction_data()
else:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
    else:
        st.info('Please upload a CSV or use sample data')
        df = None

# ---------------- Dashboard ----------------
if df is not None:
    st.subheader('Transaction Data Preview')
    st.dataframe(df.head(20))
    
    # ---------- Aggregate Trends ----------
    st.subheader('Aggregated Purchase Trends')
    trend_df = df.groupby('date').agg(
        total_purchase=('purchase_value','sum'),
        transactions=('purchase_value','count'),
        avg_purchase=('purchase_value','mean')
    ).reset_index()
    st.dataframe(trend_df)

    # ---------- Visualize Trends ----------
    st.subheader('Purchase Trend Visualization')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(trend_df['date'], trend_df['total_purchase'], marker='o', label='Total Purchase')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Purchase')
    ax.set_title('Total Purchases Over Time')
    ax.grid(True)
    st.pyplot(fig)
    
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(trend_df['date'], trend_df['transactions'], marker='x', color='orange', label='Transactions')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_title('Number of Transactions Over Time')
    ax2.grid(True)
    st.pyplot(fig2)

    # ---------- Forecast Future Purchases ----------
    st.subheader('Forecast Next 7 Days')
    X = np.arange(len(trend_df)).reshape(-1,1)
    y = trend_df['total_purchase'].values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(trend_df), len(trend_df)+7).reshape(-1,1)
    forecast = model.predict(future_X)
    future_dates = pd.date_range(trend_df['date'].max() + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({'date':future_dates, 'predicted_total_purchase':forecast.round(2)})
    st.dataframe(forecast_df)

    # Visualize forecast with historical data
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(trend_df['date'], trend_df['total_purchase'], marker='o', label='Historical')
    ax3.plot(forecast_df['date'], forecast_df['predicted_total_purchase'], marker='x', linestyle='--', color='red', label='Forecast')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Total Purchase')
    ax3.set_title('Purchase Trend Forecast')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # ---------- Download Reports ----------
    st.subheader('Download Reports')
    trend_csv = trend_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Trend Data CSV', data=trend_csv, file_name='purchase_trends.csv', mime='text/csv')
    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Forecast CSV', data=forecast_csv, file_name='purchase_forecast.csv', mime='text/csv')

st.markdown('---')
st.caption('This app helps analyze customer purchase trends and predicts future purchases for better marketing decisions.')
