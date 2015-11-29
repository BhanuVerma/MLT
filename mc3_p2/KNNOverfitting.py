"""
KNN Overfitting checkpoint.  (c) 2015 Bhanu Verma
"""

import numpy as np
import math
import KNNLearner as knn
import pandas as pd
import matplotlib.pyplot as plot


if __name__ == "__main__":
    inf = open('Data/ripple.csv')
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    total = 15
    in_sample_error = []
    out_sample_error = []
    k_values = []

    for i in range(total):
        # print "KNNLearner"
        k_values.append(i + 1)
        learner = knn.KNNLearner(k=i + 1)  # create a KNNLearner

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        # print
        # print "In sample results"
        # print "RMSE: ", rmse
        in_sample_error.append(rmse)
        c = np.corrcoef(predY, y=trainY)
        # print "corr: ", c[0, 1]

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        # print
        # print "Out of sample results"
        # print "RMSE: ", rmse
        out_sample_error.append(rmse)
        c = np.corrcoef(predY, y=testY)
        # print "corr: ", c[0, 1]
        # print

    in_sample = np.asarray(in_sample_error)
    out_sample = np.asarray(out_sample_error)

    in_sample_df = pd.DataFrame(data=in_sample, index=k_values, columns=['InSample'])
    out_sample_df = pd.DataFrame(data=out_sample, index=k_values, columns=['OutSample'])

    overfit_index = 0

    for index, row in out_sample_df.iterrows():
        current = out_sample_df.loc[index, 'OutSample']
        if index == 1:
            previous = current

        if index != 1:
            if (current / previous) > 1.0:
                overfit_index = index - 1
                break
        previous = current

    plot.figure()
    ax = plot.gca()
    ax.set_title('Overfitting')
    ax.set_xlabel('K Values')
    ax.set_ylabel('RMSE')
    out_sample_df.plot(ax=ax, color='green')
    in_sample_df.plot(ax=ax, color='blue')
    plot.axvline(overfit_index, color='orange')
    ax.legend(loc='best')
    plot.show()
