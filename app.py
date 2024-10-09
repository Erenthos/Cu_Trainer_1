import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Title of the app
st.title('Copper Price Forecasting with RandomForestRegressor')

# Load a larger dataset (replace with actual large dataset)
data = pd.read_csv('copper_price_data.csv')  # Replace with the actual dataset file

# Convert 'Month' column to datetime if needed
data['Month'] = pd.to_datetime(data['Month'])

# Extract the relevant features (replace with actual dataset structure)
data['Month_Number'] = np.arange(1, len(data) + 1)  # Numeric encoding for time

# Create more features if available
# For example, add economic indicators, supply-demand factors, etc.
# data['some_other_feature'] = ... 

# Splitting data into training and testing sets (80% train, 20% test)
X = data[['Month_Number']]  # You can add more features here
y = data['Actual_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict prices for both train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate model performance
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Display performance metrics
st.subheader('Model Performance')
st.write(f"Training MAE: {mae_train}")
st.write(f"Test MAE: {mae_test}")
st.write(f"Training R²: {r2_train}")
st.write(f"Test R²: {r2_test}")

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 12, 6)

# Predict future prices
last_month = data['Month'].iloc[-1]
future_months = [(last_month + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(1, num_months + 1)]
future_month_numbers = np.arange(len(data) + 1, len(data) + 1 + num_months).reshape(-1, 1)
predicted_future_prices = model.predict(future_month_numbers)

# Create future dataframe
future_df = pd.DataFrame({
    'Month': future_months,
    'Predicted_Price': predicted_future_prices
})

# Input: Actual values for forecasted months (optional)
st.sidebar.header('Actual Values for Forecasted Months (Optional)')
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted months, separated by commas (e.g., 750, 760)', '')

if actual_future_prices:
    actual_future_prices = [float(x) for x in actual_future_prices.split(',')]
    if len(actual_future_prices) == num_months:
        future_df['Actual_Price'] = actual_future_prices
        future_df['Variance (%)'] = ((future_df['Predicted_Price'] - future_df['Actual_Price']) / future_df['Actual_Price']) * 100
    else:
        st.error(f"Please provide {num_months} values for the forecasted months.")

# Display future prices
st.subheader('Forecasted Prices for Upcoming Months')
st.write(future_df)

# Plot actual vs predicted prices
st.subheader('Price Forecast Visualization')
fig, ax = plt.subplots(figsize=(10, 5))

# Plot historical data
ax.plot(data['Month'].dt.strftime('%b %Y'), data['Actual_Price'], label='Actual Price', marker='o')
ax.plot(data['Month'].dt.strftime('%b %Y'), model.predict(data[['Month_Number']]), label='Predicted Price', linestyle='--', marker='x')

# Plot future predicted prices
ax.plot(future_months, future_df['Predicted_Price'], label='Future Forecast', linestyle='-.', marker='s', color='red')

# Make the x-axis scalable to avoid label overlap
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Month/Year')
ax.set_ylabel('Price (INR/KG)')
ax.set_title('Actual vs Predicted Prices')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Calculate variance (error percentage) for historical data
data['Variance (%)'] = ((model.predict(data[['Month_Number']]) - data['Actual_Price']) / data['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices (Historical Data)')
st.write(data[['Month', 'Actual_Price', 'Variance (%)']])

# Display variance table for future data (if actual values were provided)
if 'Variance (%)' in future_df.columns:
    st.subheader('Variance between Actual and Predicted Prices (Forecasted Data)')
    st.write(future_df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])
