import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    centroids_indices = np.random.choice(range(X.shape[0]), k, replace=False)
    centroids = X[centroids_indices]
    
    return np.asarray(centroids).astype(np.float32) 

def minkoski_distance(X, Y, p):
    sum = np.sum(np.abs(X-Y)**p, axis=1)
    distance = sum ** (1/p)
    return distance

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    
    distances = np.zeros((k, X.shape[0]))

    for i, centroid in enumerate(centroids):
        distances[i] = minkoski_distance(X, centroid, p)
    
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    
    centroids, classes = kmeans_computation(X, k, p, centroids, max_iter)

    return centroids, classes

def kmeans_computation(X, k, p, centroids, max_iter=100):
    prev_minimal_centroids = None
    
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        minimal_centroids = np.argmin(distances, axis=0)

        if np.array_equal(minimal_centroids, prev_minimal_centroids): break

        for j in range(k):
            centroids[j] = np.mean(X[minimal_centroids == j], axis=0)
        
        prev_minimal_centroids = minimal_centroids

    classes = minimal_centroids

    return centroids, classes
    

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    
    init_centroid = X[np.random.choice(X.shape[0])]
    centroids = [init_centroid]
    
    for i in range(k - 1):
        distances = lp_distance(X, centroids, p)
        minimal_distances = np.min(distances, axis=0)
        probabilities =  minimal_distances ** 2 / np.sum(minimal_distances ** 2)
        next_centroid = np.random.choice(X.shape[0], p=probabilities)
        centroids.append(X[next_centroid])

    centroids, classes = kmeans_computation(X, k, p, centroids, max_iter)

    return np.array(centroids), classes
