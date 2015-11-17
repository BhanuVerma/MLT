"""
best4KNN.  (c) 2015 Bhanu Verma
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn

if __name__ == "__main__":

    total = 3000
    train_length = 0.6 * total
    test_length = total - train_length

    x1_data = np.random.random_sample([total])
    x1_noise = np.random.random_sample([total])
    x1_data += x1_noise

    x2_data = np.random.random_sample([total])
    x2_noise = np.random.random_sample([total])
    x2_data += x2_noise

    x3_data = np.random.random_sample([total])
    x3_noise = np.random.random_sample([total])
    x3_data += x3_noise

    y_data = x1_data ** 4 + x2_data ** 3 + x3_data ** 2

    trainX = np.ones((train_length, 3))
    trainX[:, 0] = x1_data[0:train_length]
    trainX[:, 1] = x2_data[0:train_length]
    trainX[:, 2] = x3_data[0:train_length]
    trainY = y_data[0:train_length]

    testX = np.ones((test_length, 3))
    testX[:, 0] = x1_data[train_length:total]
    testX[:, 1] = x2_data[train_length:total]
    testX[:, 2] = x3_data[train_length:total]

    testY = y_data[train_length:total]

    total = 2

    for i in range(total):
        if i == 0:
            print "LinRegLearner"
            learner = lrl.LinRegLearner()  # create a LinRegLearner
        elif i == 1:
            print "KNNLearner"
            learner = knn.KNNLearner(k=3)  # create a KNNLearner

        learner.addEvidence(trainX, trainY)  # train it

        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0, 1]

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0, 1]
        print
