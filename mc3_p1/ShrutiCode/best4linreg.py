"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

if __name__=="__main__":
    datax1 = np.random.exponential(scale=1.0,size=30000)
    datax2 = np.random.exponential(scale=1.0,size=30000)    
    datay = datax1 * np.random.rand() + datax2 * np.random.rand()# + datax3

    trainX = np.empty((18000,2))
    trainX[:,0] = datax1[0:18000]
    trainX[:,1] = datax2[0:18000]
    trainY = datay[0:18000]

    testX = np.empty((12000,2))
    testX[:,0] = datax1[18000:30000]
    testX[:,1] = datax2[18000:30000]
    testY = datay[18000:30000]

    # create a learner and train it
    print "LinRegLearner"
    learner = lrl.LinRegLearner() # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse

    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    print "KNNLearner"
    learner = knn.KNNLearner(3)
    learner.addEvidence(trainX, trainY)
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse

    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]