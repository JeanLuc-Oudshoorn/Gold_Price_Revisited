# Imports
import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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

# Create pseudo-forecast PCX based on rolling average of last 12 months
def create_pseudo_forecast(data, name='PF_PCX', original='PCX', year=2019, rolling=True):
    """
    Creates a rolling average forecast for months to be predicted on based on data available until
    the start of the forecast. (I.e.: Apr 2019 will be the average of Apr 2018 - Mar 2019,
    however Nov 2021 will be the average of Nov 2018 - Mar 2019. This is because the forecast would
    need to be made in March 2019 itself.)

    :param data: dataframe
    :param name: name of new variable
    :param original: name of original variable column in dataframe
    :return: original dataframe with new column added
    """
    if rolling:
        data[name] = data[original]
        data.loc[str(year) + '-04-01':str(year) + '-12-01', name] = None
        data[name] = data[name].transform(lambda x: x.rolling(12, 1).mean())
    else:
        data[name] = data[original]
        data.loc[str(year) + '-04-01':str(year) + '-12-01', name] = None
        data[name] = data[name].ffill()

    return data


# Create moving average variable of past prices
gold_data = create_pseudo_forecast(gold_data, name='PF_price_2019', original='price', year=2019)
gold_data = create_pseudo_forecast(gold_data, name='PF_price_2020', original='price', year=2020)
gold_data = create_pseudo_forecast(gold_data, name='PF_price_2021', original='price', year=2021)

# Add price exactly one year ago
gold_data['prev_year'] = gold_data['price'].shift(12)

# Join datasets together
economic_data = economic_data.join(gold_data.loc['2016-01-01':'2021-12-01', ['price', 'PF_price_2019', 'PF_price_2020',
                                                                             'PF_price_2021', 'prev_year']])

# Create dictionary for predictions and list for output
output = []
forecast_dict = {}

# Define function for calculating metrics
def get_mape(price, predictions):
    """
    Returns MAPE over entire series
    :param df_fc: dataframe with 'price' and 'predicted_mean' columns that has been used for forecasting
    :return: single-digit MAPE
    """
    ape = np.abs((price - predictions) / price) * 100
    ape[ape > 100] = 100

    return np.round(ape.mean(), 2)

# Fit models in a loop and save results
for year in [2019, 2020, 2021]:

    start_date = str(year) + '-04-01'
    end_date = str(year) + '-12-01'

    # Create time features
    economic_data['month'] = economic_data['Datetime'].dt.month
    economic_data['year'] = economic_data['Datetime'].dt.year

    # Split the data
    train_y = economic_data.loc[(economic_data.index < start_date)]['price']
    test_y = economic_data.loc[(economic_data.index >= start_date) &
                                   (economic_data.index <= end_date)]['price']

    train_X = economic_data.loc[(economic_data.index < start_date)] \
        [['Inflation', 'PCX', 'PF_price_2019', 'year', 'month']]
    test_X = economic_data.loc[(economic_data.index >= start_date) &
                                   (economic_data.index <= end_date)] \
        [['F_Inflation', 'F_PCX', 'PF_price_2019', 'year', 'month']]

    # Rename test features to match those of training
    test_X.rename(columns={'F_Inflation': 'Inflation',
                           'F_PCX': 'PCX'}, inplace=True)

    #######################################################################################
    #                                2. Fit ARIMA Models                                  #
    #######################################################################################

    # Instantiate Random Forest
    random_forest = RandomForestRegressor(random_state=7,
                                          n_estimators=250,
                                          verbose=False,
                                          n_jobs=-1)

    # Parameters
    params = {'max_features': [3, 4, 5, 6, 7]}

    # Perform GridSearch
    gridsearch = GridSearchCV(random_forest, params, cv=6)
    gridsearch.fit(train_X, train_y)

    #######################################################################################
    #                              3. Forecasting Apr-Dec                                 #
    #######################################################################################

    # Make predictions with random forest
    pred_y = gridsearch.predict(test_X)

    # Calculate mean absolute percent error
    mape = get_mape(test_y, pred_y)
    print("\n MAPE for {} is: {}".format(year, mape))

    # Save predictions
    forecast_dict[str(year)] = pred_y

    # Save metrics
    output.append((year, mape, gridsearch.best_params_))

#######################################################################################
#                                  4. Save Results                                    #
#######################################################################################

# Convert output to dataframe
output_df = pd.DataFrame(output, columns=['Year', 'MAPE', 'Params'])

# Save results
output_df.to_csv('metrics_rf_full.csv', index=False)

# Save dictionary with predictions as pickle file
with open('forecast_dictionary_rf_full.pkl', 'wb') as f:
    pickle.dump(forecast_dict, f)
