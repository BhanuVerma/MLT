"""
A simple wrapper for K Nearest Neighbour.  (c) 2015 Bhanu Verma
"""

import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
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
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result_y = np.zeros(points.shape[0])
        for learner in self.learners:
            temp_result = learner.query(points)
            result_y = result_y + temp_result
        result_y /= len(self.learners)
        return result_y


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
