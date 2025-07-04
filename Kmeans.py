__authors__ = ['1673296','1674485','1669906']
__group__ = '11'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_options(options)  # DICT options
        self._init_X(X)


    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """  
        npArray = np.array(X,dtype=float)
        '''      
        if(npArray.ndim == 3):
            npArray = npArray.reshape(-1,3)
        self.X = npArray
        '''
        if npArray.ndim == 3 and self.options.get('11', True):
            # Convert to 11D color descriptor
            npArray = utils.get_color_prob(npArray).reshape(-1, 11)
        elif npArray.ndim == 3:
            # Use default RGB
            npArray = npArray.reshape(-1, 3)
        
        self.X  = npArray

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if '11' not in options:
            options['11'] = False

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            punts_unics, indice = np.unique(self.X, axis=0, return_index=True)
            self.centroids = self.X[np.sort(indice)[:self.K]]
           
        elif self.options['km_init'].lower() == 'random':
            punts = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[punts]

        elif self.options['km_init'].lower() == 'custom':
            self.centroids = np.random.rand(self.K, self.X.shape[1]) * 255  # Random values between 0 and 255
        else:
            self.centroids = None

        self.old_centroids = np.zeros_like(self.centroids) if self.centroids is not None else None



    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        distancia = distance(self.X, self.centroids)
        self.labels = np.argmin(distancia, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids
        centroides = np.zeros_like(self.centroids)
        for i in range(self.centroids.shape[0]):
            grupo = self.X[self.labels == i]
            if len(grupo) > 0:
                centroides[i] = np.mean(grupo, axis=0)
            else:
                centroides[i] = self.old_centroids[i]
        self.centroids = centroides


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        self.num_iter = 0
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
            if self.converges():
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        d_total = 0.0
        for centroide in range(self.K):
            grupo = self.X[self.labels == centroide]
            if len(grupo)> 0:
                distancia = np.linalg.norm(grupo - self.centroids[centroide], axis=1)
                d_total = d_total + np.sum(distancia*distancia)
            self.WCD = d_total/len(self.X)
        return self.WCD

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        wcdAntes = None
        buscada = max_K
        for k in range(2, max_K + 1):
            kmeans = KMeans(self.X, K=k, options=self.options)
            kmeans.fit()
            wcdActual = kmeans.withinClassDistance()
            if wcdAntes is not None:
                caida = 100*(wcdAntes-wcdActual)/wcdAntes
                if caida < 20:
                    buscada = k - 1
                    break
            wcdAntes = wcdActual
        self.K = buscada


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    probs = utils.get_color_prob(centroids)
    indices = np.argmax(probs, axis = 1)
    colores =[]
    for i in indices:
        colores.append(utils.colors[i])
    return colores
    

    