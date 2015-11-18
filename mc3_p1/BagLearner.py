"""
A simple wrapper for K Nearest Neighbour.  (c) 2015 Bhanu Verma
"""

import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost):
        if bags <= 0:
            bags = 20
        if 'k' in kwargs:
            if kwargs['k'] <= 0:
                kwargs['k'] = 3
        else:
            kwargs['k'] = 3
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY):
        row_count, column_count = dataX.shape
        count = 0
        for l in self.learners:
            sampled_x = np.zeros(dataX.shape)
            sampled_indices = np.random.randint(0, row_count, size=row_count)
            for column_index in range(column_count):
                sampled_x[:, column_index] = np.array([dataX[row_index, column_index] for row_index in sampled_indices])
            sampled_y = np.array([dataY[row_index] for row_index in sampled_indices])
            l.addEvidence(sampled_x, sampled_y)
            count += 1

    def query(self, points):
        result_y = np.zeros(points.shape[0])
        for learner in self.learners:
            temp_result = learner.query(points)
            result_y = result_y + temp_result
        result_y /= len(self.learners)
        return result_y


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
