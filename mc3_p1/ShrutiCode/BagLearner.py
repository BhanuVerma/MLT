import numpy as np
import pandas as pd
class BagLearner(object):
    def __init__(self,learner,kwargs,bags,boost):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = list()
        for i in range(bags):
            self.learners.append(learner(**kwargs))

        
    def addEvidence(self,dataX,dataY):
        maxX = max(dataX[:,i].max() for i in range(dataX.shape[1]))
        minX = min(dataX[:,i].min() for i in range(dataX.shape[1]))
        rows = dataX.shape[0]
        dfX = pd.DataFrame(dataX)
        for l in self.learners:
            newX = np.zeros(dataX.shape)
            newY = np.zeros(dataX.shape[0])
            indices1 = np.random.randint(0,rows,size=0.6*rows)
            indices2 = np.random.randint(0,rows,size=0.4*rows)
            count = 0
            for i in range(dataX.shape[1]):
                aX = np.array([dataX[j,i] for j in indices1])
                bX = np.array([dataX[j,i] for j in indices2])
                newX[:,i] = np.concatenate([aX,bX])
                if count == 0:
                    print newX
                count += 1
            aY = np.array([dataY[j] for j in indices1])
            bY = np.array([dataY[j] for j in indices2])
            newY = np.concatenate([aY,bY])
            l.addEvidence(newX,newY)
    
    def query(self,dataX):
        resultsX = np.zeros(dataX.shape[0])
        for learner in self.learners:
            r = learner.query(dataX)
            resultsX = resultsX+r
        resultsX = resultsX/len(self.learners)
        return resultsX