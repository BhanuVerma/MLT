"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

if __name__=="__main__":

    datax1 = np.random.random_sample([52000])
    noisex1 = np.random.exponential(scale=1.0,size=52000)    
    datax1+=noisex1
    datax2 = np.random.random_sample([52000])
    noisex2 = np.random.exponential(scale=1.0,size=52000)    
    datax2+=noisex2    
    datax3 = np.random.random_sample([52000])
    noisex3 = np.random.exponential(scale=1.0,size=52000)
    datax3+=noisex3    
    datax4 = np.random.random_sample([52000])
    noisex4 = np.random.exponential(scale=1.0,size=52000)
    datax4+=noisex4    
    datax5 = np.random.random_sample([52000])
    noisex5 = np.random.exponential(scale=1.0,size=52000)
    datax5+=noisex5    
    datay = datax1 * np.random.rand() + datax2 * np.random.rand() + datax3 * np.random.rand() + datax4 * np.random.rand() + datax5 * np.random.rand()

    trainX = np.ones((31200,5))
    trainX[:,0] = datax1[0:31200]
    trainX[:,1] = datax2[0:31200]
    trainX[:,2] = datax2[0:31200]
    trainX[:,3] = datax2[0:31200]
    trainX[:,4] = datax2[0:31200]
    trainY = datay[0:31200]

    testX = np.ones((20800,5))
    testX[:,0] = datax1[31200:52000]
    testX[:,1] = datax2[31200:52000]
    testX[:,2] = datax2[31200:52000]
    testX[:,3] = datax2[31200:52000]
    testX[:,4] = datax2[31200:52000]
    testY = datay[31200:52000]

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