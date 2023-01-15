# Imports
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Read & import gold price data
gold_data = pd.read_csv("gold_price.csv")

# Create date column
gold_data['Datetime'] = pd.to_datetime(gold_data['date'], format='%Y-%m-%d')

# Set date as index
gold_data.set_index(gold_data['Datetime'], inplace=True)

# Load forecasts
with open('forecast_dictionary_full.pkl', 'rb') as f:
    forecasts = pickle.load(f)

# Plot forecast for all three years
fig, axs = plt.subplots(3, 1, figsize=(9, 12))

# Forecast for 2019
f2019 = forecasts['2019']

axs[0].plot(gold_data['2016-01-01':'2019-12-01'].index, gold_data.loc['2016-01-01':'2019-12-01', 'price'],
            label='observed')
axs[0].plot(f2019.index, f2019['predicted_mean'], color='r', label='forecast')
axs[0].fill_between(f2019.index, f2019['lower price'], f2019['upper price'], color='pink')
axs[0].set_title("Gold Price Forecast: Apr-Dec 2019")
axs[0].set_ylim((0, f2019['upper price'].max()*1.1))

# Forecast for 2020
f2020 = forecasts['2020']

axs[1].plot(gold_data['2016-01-01':'2020-12-01'].index, gold_data.loc['2016-01-01':'2020-12-01', 'price'],
            label='observed')
axs[1].plot(f2020.index, f2020['predicted_mean'], color='r', label='forecast')
axs[1].fill_between(f2020.index, f2020['lower price'], f2020['upper price'], color='pink')
axs[1].set_title("Gold Price Forecast: Apr-Dec 2020")
axs[1].set_ylim((0, f2020['upper price'].max()*1.1))

# Forecast for 2020
f2021 = forecasts['2021']

axs[2].plot(gold_data['2016-01-01':'2021-12-01'].index, gold_data.loc['2016-01-01':'2021-12-01', 'price'],
            label='observed')
axs[2].plot(f2021.index, f2021['predicted_mean'], color='r', label='forecast')
axs[2].fill_between(f2021.index, f2021['lower price'], f2021['upper price'], color='pink')
axs[2].set_title("Gold Price Forecast: Apr-Dec 2021")
axs[2].set_ylim((0, f2021['upper price'].max()*1.1))

# Save and show plot
plt.savefig('gold_price_predictions.png', bbox_inches='tight')
plt.show()
