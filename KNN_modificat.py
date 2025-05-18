__authors__ = ['1673296','1674485','1669906']
__group__ = '11'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels, metric='euclidean'):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.metric = metric

    def _init_train(self, train_data):
        data = np.array(train_data, dtype=float)
        self.train_data = data.reshape(train_data.shape[0], -1)

    def get_k_neighbours(self, test_data, k):
        datos = np.array(test_data, dtype=float)
        prueba = datos.reshape(test_data.shape[0], -1)

        distancias = cdist(prueba, self.train_data, metric=self.metric)

        indices = np.argsort(distancias, axis=1)[:, :k]
        self.neighbors = self.labels[indices]

    def get_class(self):
        preds = []
        for i in range(self.neighbors.shape[0]):
            pred = max(self.neighbors[i], key=lambda x: (np.sum(self.neighbors[i] == x), -np.where(self.neighbors[i] == x)[0][0]))
            preds.append(pred)
        return np.array(preds)

    def predict(self, test_data, k):
        self.get_k_neighbours(test_data, k)
        return self.get_class()
