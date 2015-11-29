"""
Bagging effect.  (c) 2015 Bhanu Verma
"""

import numpy as np
import math
import KNNLearner as knn
import pandas as pd
import matplotlib.pyplot as plot
import BagLearner as bl


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
        k_values.append(i + 1)
        learner = bl.BagLearner(learner=knn.KNNLearner, kwargs={"k": i+1}, bags=20, boost=False)  # create a BagLearner

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        # print
        # print "In sample results"
        # print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        in_sample_error.append(rmse)
        # print "corr: ", c[0, 1]

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        # print
        # print "Out of sample results"
        # print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        out_sample_error.append(rmse)
        # print "corr: ", c[0, 1]
        # print

    in_sample = np.asarray(in_sample_error)
    out_sample = np.asarray(out_sample_error)

    in_sample_df = pd.DataFrame(data=in_sample, index=k_values, columns=['InSample'])
    out_sample_df = pd.DataFrame(data=out_sample, index=k_values, columns=['OutSample'])

    overfit_index = 0

    # for index, row in out_sample_df.iterrows():
    #     current = out_sample_df.loc[index, 'OutSample']
    #     if index == 1:
    #         previous = current
    #
    #     if index != 1:
    #         if (current / previous) > 1.0:
    #             overfit_index = index - 1
    #             break
    #     previous = current

    plot.figure()
    ax = plot.gca()
    ax.set_title('Overfitting')
    ax.set_xlabel('K Values')
    ax.set_ylabel('RMSE')
    out_sample_df.plot(ax=ax, color='green')
    in_sample_df.plot(ax=ax, color='blue')
    ax.legend(loc='best')
    plot.show()
