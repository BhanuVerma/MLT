"""
Test a learner.  (c) 2015 Bhanu Verma
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

if __name__=="__main__":
    datax1 = np.random.exponential(scale=1.0, size=3000)
    datax2 = np.random.exponential(scale=1.0, size=3000)
    datay = datax1 * np.random.rand() + datax2 * np.random.rand()

    trainX = np.empty((1800, 2))
    trainX[:, 0] = datax1[0:1800]
    trainX[:, 1] = datax2[0:1800]
    trainY = datay[0:1800]

    testX = np.empty((1200, 2))
    testX[:, 0] = datax1[1800:3000]
    testX[:, 1] = datax2[1800:3000]
    testY = datay[1800:3000]

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
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0, 1]

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0, 1]
        print