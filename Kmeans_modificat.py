__authors__ = ['1673296','1674485','1669906']
__group__ = '11'

import numpy as np
import utils
import matplotlib as plt

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
        

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """  
        npArray = np.array(X, dtype=float)

        if npArray.ndim == 3 and self.options['11'] == True:
            # Covierte espacio 11 dimensiones
            npArray = utils.get_color_prob(npArray).reshape(-1, 11)
        elif npArray.ndim == 3:
            # Usa RGB normal
            npArray = npArray.reshape(-1, 3)

        self.X = npArray   


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

        elif self.options['km_init'].lower() == 'farthest':
            self._init_farthest_first()
        
        elif self.options['km_init'].lower() == 'uniform':
            self._init_uniform()

        else:
            self.centroids = None

        self.old_centroids = np.zeros_like(self.centroids) if self.centroids is not None else None

    def _init_uniform(self):
        dim = self.X.shape[1]  # Dimensionalidad de los datos (ej: 3 para RGB)
        max_bins = int(np.ceil(self.K ** (1/dim)))  # Divisiones por eje
        edges = [np.linspace(0, 255, max_bins + 1) for _ in range(dim)]
        theoretical_centers = np.stack(np.meshgrid(*[(e[:-1] + e[1:])/2 for e in edges]), axis=-1).reshape(-1, dim)
        
        centroids = []
        for center in theoretical_centers:
            distances = np.linalg.norm(self.X - center, axis=1)
            closest_idx = np.argmin(distances)
            centroids.append(self.X[closest_idx])
        
        # Si hay menos celdas que K, completar con puntos aleatorios
        if len(centroids) < self.K:
            extra = self.X[np.random.choice(self.X.shape[0], self.K - len(centroids), replace=False)]
            centroids.extend(extra)
        
        self.centroids = np.array(centroids[:self.K])  # Asegurar K centroides

    def _init_farthest_first(self):
        '''
        """Secuencial Lejano (Farthest-First)"""
        centroids = [self.X[np.random.choice(self.X.shape[0])]]
        for _ in range(1, self.K):
            distances = distance(self.X, np.array(centroids))
            min_distances = np.min(distances, axis=1)
            farthest_idx = np.argmax(min_distances)
            centroids.append(self.X[farthest_idx])
        self.centroids = np.array(centroids)
        '''
        # Seleccionar el primer centroide aleatoriamente
        first_idx = np.random.choice(self.X.shape[0])
        centroids = [self.X[first_idx]]
        
        # Seleccionar los K-1 centroides restantes
        for _ in range(1, self.K):
            # Calcular distancias de todos los puntos a los centroides actuales
            distances = np.zeros((self.X.shape[0], len(centroids)))
            
            for i, centroid in enumerate(centroids):
                # Calcular distancia euclidiana para cada dimensión
                distances[:, i] = np.sqrt(np.sum((self.X - centroid) ** 2, axis=1))
            
            # Para cada punto, encontrar la distancia mínima a cualquier centroide
            min_distances = np.min(distances, axis=1)
            
            # Seleccionar el punto más lejano como nuevo centroide
            farthest_idx = np.argmax(min_distances)
            centroids.append(self.X[farthest_idx])
        
        self.centroids = np.array(centroids)


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
    
    def interClass(self):
        """
         Devuelve distancia interclass
        """
        d_total = 0.0
        conteo = 0
        for i in range(self.K):
            for j in range(i+1, self.K):
                d_total += np.linalg.norm(self.centroids[i] - self.centroids[j])
                conteo += 1

        if conteo == 0:
            return 0.0
            
        return d_total/conteo
    
    def fisher(self):
        Intra = self.withinClassDistance()
        Inter = self.interClass()
        if Inter < 1e-10:
            return float('inf') 
        self.disFish = Intra/Inter
        return self.disFish


    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        if self.options['fitting'].lower() == 'wcd':
            wcdAntes = None
            self.WCD = []
            buscada = max_K
            for k in range(2, max_K + 1):
                kmeans = KMeans(self.X, K=k, options=self.options)
                kmeans.fit()
                wcdActual = kmeans.withinClassDistance()
                self.WCD.append(wcdActual)
                if wcdAntes is not None:
                    caida = 100*(wcdAntes-wcdActual)/wcdAntes
                    if caida < 80:
                        buscada = k - 1
                        break
                self.centroids = kmeans.centroids
                wcdAntes = wcdActual
            self.K = buscada

        if self.options['fitting'].lower() == 'inter':
            self.Inter = []  # Asegúrate de que esté vacía al inicio
            best_k = max_K
            interAntes = None

            for k in range(2, max_K + 1):
                kmeans = KMeans(self.X, K=k, options=self.options)
                kmeans.fit()
                interActual = kmeans.interClass()
                self.Inter.append(interActual)

                if interAntes is not None:
                    mejora = 100 * (interActual - interAntes) / interAntes
                    if mejora < 80:
                        best_k = k - 1
                        break
                self.centroids = kmeans.centroids
                interAntes = interActual

            self.K = best_k
        
        if self.options['fitting'].lower() == 'fisher':
            fisherAntes = None
            buscada = max_K
            self.fish = []
            for k in range(2, max_K + 1):
                kmeans = KMeans(self.X, K=k, options=self.options)
                kmeans.fit()
                fisherActual = kmeans.fisher()
                self.fish.append(fisherActual)
                if fisherAntes is not None:
                    caida = 100*(fisherAntes-fisherActual)/fisherAntes
                    if caida < 80:
                        buscada = k - 1
                        break
                self.centroids = kmeans.centroids
                fisherAntes = fisherActual
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
    


def get_colors(centroids, Dim):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    if Dim == False:
        probs = utils.get_color_prob(centroids)
    else:
        probs = centroids
    indices = np.argmax(probs, axis = 1)
    colores =[]
    for i in indices:
        colores.append(utils.colors[i])
    return colores
    

