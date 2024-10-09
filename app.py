import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title('Copper Price Forecasting with Variance')

# Load a larger dataset (replace with actual data source)
# Example: df = pd.read_csv('copper_prices.csv')  # Assuming you have a CSV file with copper prices
data = {
    'Month': pd.date_range(start='2020-01-01', end='2023-07-01', freq='M'),
    'Actual_Price': np.random.uniform(low=600, high=800, size=42)  # Random data for illustration
}
df = pd.DataFrame(data)

# Feature Engineering
df['Month_Number'] = np.arange(1, len(df) + 1)
df['Lagged_Price_1'] = df['Actual_Price'].shift(1)
df['Lagged_Price_2'] = df['Actual_Price'].shift(2)
df['Moving_Average'] = df['Actual_Price'].rolling(window=3).mean()

# Drop rows with NaN values caused by lagging and rolling calculations
df.dropna(inplace=True)

# Split the data into features and target variable
X = df[['Month_Number', 'Lagged_Price_1', 'Lagged_Price_2', 'Moving_Average']]
y = df['Actual_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict prices for the test set
df['Predicted_Price'] = model.predict(X)

# Calculate MSE for evaluation
mse = mean_squared_error(y_test, model.predict(X_test))
st.write(f'Mean Squared Error: {mse:.2f}')

# Input: Number of months to forecast
st.sidebar.header('Forecast Settings')
num_months = st.sidebar.slider('Select number of months to forecast:', 1, 12, 3)

# Generate future month labels
last_month = df['Month'].iloc[-1]  # Get the last month in the DataFrame
future_months = [(last_month + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(1, num_months + 1)]

# Prepare future data for predictions
future_month_numbers = np.arange(len(df) + 1, len(df) + 1 + num_months).reshape(-1, 1)

# Lagged values for future predictions
future_lagged_prices = np.array([df['Actual_Price'].iloc[-1]] * num_months)  # Last known price for lagging
future_moving_average = np.mean(df['Actual_Price'].tail(3))  # Simple moving average for the last known prices

# Predict future prices
future_X = np.hstack((future_month_numbers, future_lagged_prices.reshape(-1, 1), future_lagged_prices.reshape(-1, 1), np.full((num_months, 1), future_moving_average)))
predicted_future_prices = model.predict(future_X)

# Create future dataframe
future_df = pd.DataFrame({
    'Month': future_months,
    'Predicted_Price': predicted_future_prices
})

# Input: Actual values for forecasted months (optional)
st.sidebar.header('Actual Values for Forecasted Months (Optional)')
actual_future_prices = st.sidebar.text_input('Enter actual prices for forecasted months, separated by commas (e.g., 750, 760)', '')

# Process the actual prices if provided
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
ax.plot(df['Month'].dt.strftime('%b %Y'), df['Actual_Price'], label='Actual Price', marker='o')
ax.plot(df['Month'].dt.strftime('%b %Y'), df['Predicted_Price'], label='Predicted Price', linestyle='--', marker='x')

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
df['Variance (%)'] = ((df['Predicted_Price'] - df['Actual_Price']) / df['Actual_Price']) * 100

# Display variance table
st.subheader('Variance between Actual and Predicted Prices (Historical Data)')
st.write(df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])

# Display variance table for future data (if actual values were provided)
if 'Variance (%)' in future_df.columns:
    st.subheader('Variance between Actual and Predicted Prices (Forecasted Data)')
    st.write(future_df[['Month', 'Actual_Price', 'Predicted_Price', 'Variance (%)']])
