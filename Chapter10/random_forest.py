import numpy as np
import pandas as pd

from simulator import simulate
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(123456)
lr = RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=5)

data = pd.read_csv('BTC-USD.csv')
data = data.dropna()
data.Date = pd.to_datetime(data.Date)
data.set_index('Date', drop=True, inplace=True)
diffs = (data.Close.diff()/data.Close).values[1:]

diff_len = len(diffs)



def create_x_data(lags=1):
    diff_data = np.zeros((diff_len, lags))
    ma_data = np.zeros((diff_len, lags))

    diff_ma = (data.Close.diff()/data.Close).rolling(15).mean().fillna(0).values[1:]
    for lag in range(1, lags+1):
        this_data = diffs[:-lag]
        diff_data[lag:, lag-1] = this_data

        this_data = diff_ma[:-lag]
        ma_data[lag:, lag-1] = this_data
    return np.concatenate((diff_data, ma_data), axis=1)

x_data = create_x_data(lags=30)*100
y_data = diffs*100

# REPRODUCIBILITY
x_data = np.around(x_data, decimals=8)
y_data = np.around(y_data, decimals=8)

# =============================================================================
# WALK FORWARD
# =============================================================================

window = 150
preds = np.zeros(diff_len-window)
for i in range(diff_len-window-1):
    x_train = x_data[i:i+window, :]
    y_train = y_data[i:i+window]
    lr.fit(x_train, y_train)
    preds[i] = lr.predict(x_data[i+window+1, :].reshape(1, -1))


print('Percentages MSE: %.2f'%metrics.mean_squared_error(y_data[window:], preds))
simulate(data, preds)



