"""
A simple wrapper for K Nearest Neighbour.  (c) 2015 Bhanu Verma
"""

import numpy as np


class KNNLearner(object):
    def __init__(self, k):
        if k <= 0:
            k = 3
        self.n_n = k
        self.x_data = None
        self.y_data = None

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # load data into class variables
        self.x_data = dataX
        self.y_data = dataY

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        row_count, column_count = points.shape
        result_y = np.zeros(row_count)
        for count in range(row_count):
            a = points[count]
            euc_dist = np.sqrt(((a - self.x_data) ** 2).sum(axis=1))
            sorted_indices = euc_dist.argsort()
            n_neighbours = sorted_indices[:self.n_n]
            nearest_y = [self.y_data[index] for index in n_neighbours]
            result_y[count] = np.mean(nearest_y)

        return result_y


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
