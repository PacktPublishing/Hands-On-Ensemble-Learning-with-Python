import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics


def simulate(data, preds):
    # Constants and placeholders
    buy_threshold = 0.5
    stake = 100

    true, pred, balances = [], [], []

    buy_price = 0
    buy_points, sell_points = [], []
    balance = 0

    start_index = len(data)-len(preds)-1
    # Calcualte predicted values
    for i in range(len(preds)):

        last_close = data.Close[i+start_index-1]
        current_close = data.Close[i+start_index]

        # Save predicted values and true values
        true.append(current_close)
        pred.append(last_close*(1+preds[i]/100))


        # Buy/Sell according to signal
        if preds[i] > buy_threshold and buy_price == 0:
            buy_price = true[-1]
            buy_points.append(i)

        elif preds[i] < -buy_threshold and not buy_price == 0:
            profit = (current_close - buy_price) * stake/buy_price
            balance += profit
            buy_price = 0
            sell_points.append(i)

        balances.append(balance)


    true = np.array(true)
    pred = np.array(pred)

    # Create plots
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(true, label='True')
    plt.plot(pred, label='pred')
    plt.scatter(buy_points, true[buy_points]+500, marker='v',
                c='blue', s=5, zorder=10)
    plt.scatter(sell_points, true[sell_points]-500, marker='^'
                , c='red', s=5, zorder=10)
    plt.title('Trades')

    plt.subplot(2, 1, 2)
    plt.plot(balances)
    plt.title('Profit')
    print('MSE: %.2f'%metrics.mean_squared_error(true, pred))
    balance_df = pd.DataFrame(balances)

    pct_returns = balance_df.diff()/stake
    pct_returns = pct_returns[pct_returns != 0].dropna()


    print('Sharpe: %.2f'%(np.mean(pct_returns)/np.std(pct_returns)))