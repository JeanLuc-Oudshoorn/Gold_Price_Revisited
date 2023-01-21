# Imports
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import seaborn as sns
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#######################################################################################
#                             1. Read and Preprocess Data                             #
#######################################################################################

# Read & import economic data
economic_data = pd.read_csv("0113_cleandata.csv")
economic_data = economic_data.loc[(economic_data['ProductGroup'] == 'A') & (economic_data['Country'] == 'USA')] \
    [['Datetime', 'IPX', 'F_IPX', 'Inflation', 'F_Inflation', 'GDP', 'F_GDP', 'PCX', 'F_PCX']]

# Recreate full date
economic_data['Datetime'] = pd.to_datetime(economic_data['Datetime'], format='%Y-%m-%d')

# Set index
economic_data.set_index('Datetime', inplace=True)

# Pre-process and join all metal dataframes to the economic data
for metal in ['gold', 'silver', 'copper', 'aluminum', 'nickel']:
    # Read CSV-file
    tmp = pd.read_csv('{}_price.csv'.format(metal))
    # Rename necessary columns
    tmp.rename(columns={'Unnamed: 0': 'Datetime',
                        'price': '{}_price'.format(metal)}, inplace=True)
    # Convert date to datetime object
    tmp['Datetime'] = pd.to_datetime(tmp['Datetime'], format='%Y-%m-%d')
    # Set datetime as index
    tmp.set_index('Datetime', inplace=True)
    # Subset the data
    tmp = tmp.loc['2014-12-01':'2021-12-01', '{}_price'.format(metal)]
    # Assign dataframe to dictionary
    economic_data = economic_data.join(tmp)

#######################################################################################
#                            2. Exploratory Data Analysis                             #
#######################################################################################

# Define variables for VARMAX
endog = economic_data[['gold_price', 'silver_price', 'copper_price', 'aluminum_price', 'nickel_price']]
exog = economic_data[['IPX', 'PCX', 'Inflation']].dropna()
exog_forecast = economic_data.loc['2019-04-01':'2019-12-01', ['F_IPX', 'F_PCX', 'F_Inflation']]

# Plot time series
# endog.plot(subplots=True, figsize=(13, 8))

# Plot pairwise correlations
# g = sns.clustermap(endog.corr(), annot=True)
# ax = g.ax_heatmap
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.title('Pairwise Correlations')
# plt.show()


#######################################################################################
#                            3. Granger's Causality Test                              #
#######################################################################################

maxlag = 12
test = 'ssr_chi2test'


# Define function to test for Granger's causation
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# Plot Granger's causation matrix
# grangers_df = grangers_causation_matrix(endog, variables = endog.columns)
#
# plt.figure(figsize=(8, 5))
# sns.heatmap(grangers_df, annot=True)
# b, t = plt.ylim()
# b += 0.5
# t -= 0.5
# plt.ylim(b, t)
# plt.show()

# Transform for stationarity
endog_transformed = endog.apply(lambda x: x.diff()).dropna()

# Define test for unit root based on Augmented Dickey-Fuller test
def test_unit_root(df):
    return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')

# Perform test
print(test_unit_root(endog_transformed))

#######################################################################################
#                                4. Fit VARMAX Models                                 #
#######################################################################################

# Create dictionary for test results
test_results = {}

# Fit models in a loop and save results
year = 2019

# Subset data for training
start_date = str(year) + '-04-01'
end_date = str(year) + '-12-01'

train_endog = endog_transformed.loc[(endog_transformed.index < start_date)]
pred_endog = endog_transformed.loc[(endog_transformed.index >= start_date) &
                                   (endog_transformed.index <= end_date)]

# Test different values for p and q
for p in range(3):
    for q in range(3):
        if p == 0 and q == 0:
            continue

        print(f'Testing Order: p = {p}, q = {q}')
        convergence_error = stationarity_error = 0

        try:
            model = VARMAX(train_endog, order=(p, q))
            model_result = model.fit(maxiter=1000, disp=False)

        except LinAlgError:
            convergence_error += 1

        except ValueError:
            stationarity_error += 1

        print('\nAIC:', model_result.aic)
        print('BIC:', model_result.bic)
        print('HQIC:', model_result.hqic)
        print('------------------------')

        test_results[(p, q)] = [model_result.aic,
                                model_result.bic,
                                convergence_error,
                                stationarity_error]


# Refit the best model
model = VARMAX(train_endog, order=(1, 0)).fit(maxiter=1000)

#######################################################################################
#                              5. Forecast into Future                                #
#######################################################################################

# Generate forecast with best model
forecast = model.forecast(start_date=start_date, dynamic=True, steps=9)
