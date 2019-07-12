import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the data, parse the dates and set the dates as index
data = pd.read_csv('BTC-USD.csv')
data = data.dropna()
data.Date = pd.to_datetime(data.Date)
data.set_index('Date', drop=True, inplace=True)


# =============================================================================
# ORIGINAL
# =============================================================================
# Plot ACF-> Non-Stationary
plot_acf(data.Close, lags=30)
plt.xlabel('Date')
plt.ylabel('Correlation')

# =============================================================================
# Percentage Differences
# =============================================================================

# Make two subplots
fig, axes = plt.subplots(nrows=2, ncols=1)

# Calculate the percentage differences
diffs = data.Close.diff()/data.Close

# Plot the rolling deviation
diffs.rolling(30).std().plot(ax=axes[0])
plt.xlabel('Date')
plt.ylabel('Std. Dev.')
axes[0].title.set_text('Transformed Data Rolling Std.Dev.')

diffs = diffs.dropna()

# Plot ACF for percentage diffs
plot_acf(diffs, lags=60, ax=axes[1])
plt.xlabel('Date')
plt.ylabel('Correlation')

# Plot the changes
plt.figure()
diffs.plot()
plt.xlabel('Date')
plt.ylabel('Change %')
plt.title('Transformed Data')




