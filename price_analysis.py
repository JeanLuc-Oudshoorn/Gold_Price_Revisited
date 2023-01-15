# Imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#######################################################################################
#                             1. Read and Preprocess Data                             #
#######################################################################################

# Read & import gold price data
gold_data = pd.read_csv("gold_price.csv")

# Read & import economic data
economic_data = pd.read_csv("0113_cleandata.csv")
economic_data = economic_data.loc[(economic_data['ProductGroup'] == 'A') & (economic_data['Country'] == 'USA')] \
    [['Datetime', 'IPX', 'F_IPX', 'Inflation', 'F_Inflation', 'GDP', 'F_GDP', 'PCX', 'F_PCX']]

# Recreate full date
economic_data['Datetime'] = pd.to_datetime(economic_data['Datetime'], format='%Y-%m-%d')
gold_data['Datetime'] = pd.to_datetime(gold_data['date'], format='%Y-%m-%d')

# Set date as index
economic_data.set_index(economic_data['Datetime'], inplace=True)
gold_data.set_index(gold_data['Datetime'], inplace=True)

# Join datasets together
economic_data = economic_data.join(gold_data.loc['2016-01-01':'2021-12-01', 'price'])

# Define function for calculating metrics
def get_mape(df_fc):
    """
    Returns MAPE over entire series
    :param df_fc: dataframe with 'price' and 'predicted_mean' columns that has been used for forecasting
    :return: single-digit MAPE
    """
    ape = np.abs((df_fc['price'] - df_fc['predicted_mean']) / df_fc['price']) * 100
    ape[ape > 100] = 100

    return np.round(ape.mean(), 2)


# Pre-specify a list for output and dictionary for predictions
output = []
forecast_dict = {}

# Fit models in a loop and save results
for year in [2019, 2020, 2021]:

    start_date = str(year) + '-04-01'
    end_date = str(year) + '-12-01'

    train_endog = economic_data.loc[(economic_data.index < start_date)]
    pred_endog = economic_data.loc[(economic_data.index >= start_date) &
                                   (economic_data.index <= end_date)]

    train_exog = train_endog[['Inflation', 'GDP', 'PCX']]
    pred_exog = pred_endog[['F_Inflation', 'F_GDP', 'F_PCX']]

    # Check for stationarity
    adf = adfuller(train_endog['price'].diff().dropna())

    #######################################################################################
    #                                2. Fit ARIMA Models                                  #
    #######################################################################################

    # Create a list for model results
    order_aic_bic = []

    # Fit models in a loop
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        model = SARIMAX(train_endog['price'], exog=train_exog,
                                        order=(p, 1, q),
                                        seasonal_order=(P, 1, Q, 12))
                        result = model.fit(warn_convergence=False, disp=False)

                        order_aic_bic.append((p, q, P, Q, result.aic, result.bic, result.mae))

                    except:
                        order_aic_bic.append((p, q, P, Q, None, None, None))

    order_df = pd.DataFrame(order_aic_bic,
                            columns=['p', 'q', 'P', 'Q', 'AIC', 'BIC', 'MAE'])

    # Print best parameters
    print(order_df.sort_values('AIC'))

    # Best parameters based on AIC
    best_params = order_df.sort_values('AIC').head(1)[['p', 'q', 'P', 'Q']].squeeze()
    print("\n", best_params)

    # Retrain best model
    model = SARIMAX(train_endog['price'], exog=train_exog,
                    order=(best_params[0], 1, best_params[1]),
                    seasonal_order=(best_params[2], 1, best_params[3], 12))

    result = model.fit(warn_convergence=False, disp=False)

    # Print and save coefficients and P-values
    print(result.summary())

    with open('model_summary_{}.txt'.format(year), 'w') as f:
        f.write(result.summary().as_text())

    #######################################################################################
    #                              3. Forecasting Apr-Dec                                 #
    #######################################################################################

    # Make forecast
    pred = result.get_prediction(start=start_date, exog=pred_exog, dynamic=True, end=end_date)

    # Extract mean forecast
    mean_forecast = pred.predicted_mean

    # Extract confidence intervals
    confidence_intervals = pred.conf_int()

    lower_limit = confidence_intervals.loc[:, 'lower price']
    upper_limit = confidence_intervals.loc[:, 'upper price']

    # Join forecast to actuals
    pred_endog = pred_endog.join(mean_forecast).join(lower_limit).join(upper_limit)

    # Print MAPE
    mape = get_mape(pred_endog)
    print("\n MAPE is:", mape)

    # Save results
    output.append((year, mape, best_params[0], best_params[1], best_params[2], best_params[3]))

    forecast_dict[str(year)] = pred_endog

#######################################################################################
#                                  4. Save Results                                    #
#######################################################################################

# Convert output to dataframe
output_df = pd.DataFrame(output, columns=['Year', 'MAPE', 'p', 'q', 'P', 'Q'])

# Save results
output_df.to_csv('metrics_full.csv', index=False)

# Save dictionary with predictions as pickle file
with open('forecast_dictionary_full.pkl', 'wb') as f:
    pickle.dump(forecast_dict, f)
