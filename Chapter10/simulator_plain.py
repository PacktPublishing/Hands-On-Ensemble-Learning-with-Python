import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics


def simulate(data, preds):
    true, pred= [], []

    start_index = len(data)-len(preds)-1
    for i in range(len(preds)):

        last_close = data.Close[i+start_index-1]
        current_close = data.Close[i+start_index]

        true.append(current_close)
        pred.append(last_close*(1+preds[i]/100))





    true = np.array(true)
    pred = np.array(pred)

    plt.figure()

    plt.plot(true, label='True')
    plt.plot(pred, label='pred')

    print('MSE: %.2f'%metrics.mean_squared_error(true, pred))
