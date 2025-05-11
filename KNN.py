__authors__ = ['1673296','1674485','1669906']
__group__ = '11'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        data = np.array(train_data, dtype=float)
        self.train_data = data.reshape(train_data.shape[0], -1)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        datos = np.array(test_data, dtype=float)
        prueba = datos.reshape(test_data.shape[0], -1)
        distancias = cdist(prueba,self.train_data,metric='euclidean')
        indices = np.argsort(distancias, axis=1)[:, :k]
        self.neighbors = self.labels[indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        preds = []
        for i in range(self.neighbors.shape[0]):
            pred = max(self.neighbors[i], key=lambda x: (np.sum(self.neighbors[i] == x), -np.where(self.neighbors[i] == x)[0][0]))
            preds.append(pred)
        return np.array(preds)


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()