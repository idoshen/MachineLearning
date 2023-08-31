import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []
    
    def sigmoid(self, X):
      denominator = (1 + np.exp(np.dot(X, -(np.transpose(self.theta)))))
      sigmoid = 1 / denominator
      return sigmoid

    def costFunction(self, X, y):

      numerator = (-(np.dot(y, np.log(self.sigmoid(X)))) - np.dot((1 - y), np.log(1 - self.sigmoid(X))))
      fraction = numerator / len(y)
      
      return fraction

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        sum = 0
        self.theta = np.random.rand(X.shape[1] + 1)
        dataWithBias = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for i in range(self.n_iter):
          self.Js.append(self.costFunction(dataWithBias, y))
          self.thetas.append(self.theta)
          self.theta = self.theta - self.eta * np.dot(dataWithBias.T, self.sigmoid(dataWithBias) - y)
          if (i > 0) and (self.Js[-2] - self.Js[-1] < self.eps): break 
            

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        dataWithBias = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        preds = self.sigmoid(dataWithBias)

        preds[preds < 0.5] = 0
        preds[preds >= 0.5] = 1

        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    fold_size = len(X) // folds

    fold_indices = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(folds)]
    accuracies = []

    for fold in fold_indices:
        X_train = np.delete(X_shuffled, fold, axis=0)
        y_train = np.delete(y_shuffled, fold)
        X_test = X_shuffled[fold]
        y_test = y_shuffled[fold]

        algo.fit(X_train, y_train)
        pred = algo.predict(X_test)

        accuracy = np.mean(pred == y_test)
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
 
    numerator = np.exp(-0.5 * ((data - mu.T) / sigma)**2)
    denominator = sigma * np.sqrt(2 * np.pi)
    p = numerator / denominator
    
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.costs = [] 

        self.weights = np.random.rand(self.k)
        self.weights /= np.sum(self.weights)
        max = np.max(data)
        min = np.min(data)
        self.mus = np.random.uniform(min, max, self.k)
        self.sigmas = np.random.rand(self.k)
        self.responsibilities = []
        
    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        numerator = self.weights * norm_pdf(data, self.mus, self.sigmas)
        denominator = np.sum(numerator, axis = 1, keepdims = True)
        self.responsibilities = (numerator / denominator)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.sum(self.responsibilities, axis = 0) / data.shape[0]
        self.mus = np.sum(self.responsibilities * data, axis = 0) / (self.weights * data.shape[0])
        self.sigmas = np.sqrt(np.sum(self.responsibilities * ((data - self.mus)**2), axis = 0) / (self.weights * data.shape[0]))

        
    def logLikelihood(self, X):
        """ 
        The cost of one gaussian
        """
        return np.sum(-np.log2(self.weights * norm_pdf(X, self.mus, self.sigmas)))

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)

        for i in range(self.n_iter):
          self.expectation(data)
          self.maximization(data)
          cost = self.logLikelihood(data)
          self.costs.append(cost)
          if (i > 0) and (np.abs(self.costs[-2] - self.costs[-1]) < self.eps): break 

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None

    pdf = np.sum(weights * norm_pdf(data, mus, sigmas))

    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.priors = []
        self.classes = None
        self.ems = []

    def class_update(self, y):
        labels, numberOfAppearances = np.unique(y , return_counts = True)
        self.classes = labels.tolist()
        for i in range(len(self.classes)):
          self.priors.append(numberOfAppearances[i] / len(y))
        
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.class_update(y)

        for label in self.classes:
          labeledData = X[y == label]
          for feature in labeledData.T:
            em = EM(k = self.k, random_state=self.random_state)
            em.fit(feature.reshape(-1, 1))
            self.ems.append(em)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        for instance in X:
          posteriors = []
          for index in self.classes:
            likelihood = 1
            for i, em in enumerate(self.ems[index * len(instance): (index + 1) * len(instance)]):
              weights, mus, sigmas = em.get_dist_params()
              likelihood *= gmm_pdf(instance[i], weights, mus, sigmas)
            posterior = likelihood * self.priors[index]
            posteriors.append(posterior)
          preds.append(self.classes[np.argmax(posteriors)])

        return np.array(preds)
        
def accuracy(X , y):
    return np.count_nonzero(X == y) / len(y)

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lr = LogisticRegressionGD(eta = best_eta , eps = best_eps)
    lr.fit(x_train , y_train)

    lor_train_acc = accuracy(lr.predict(x_train) , y_train)
    lor_test_acc = accuracy(lr.predict(x_test) , y_test)

    nb = NaiveBayesGaussian(k)
    nb.fit(x_train , y_train)

    bayes_train_acc = accuracy(nb.predict(x_train) , y_train)
    bayes_test_acc = accuracy(nb.predict(x_test) , y_test)
    
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None


    # Dataset A - Not linearly Seperable
    # Group 1: Positive x, positive y, positive z corner
    group_1_features = np.random.uniform(low=[0.5, 0.5, 0.5], high=[1, 1, 1], size=(250, 3))
    group_1_labels = np.zeros(250)

    # Group 2: Negative x, positive y, positive z corner
    group_2_features = np.random.uniform(low=[-1, 0.5, 0.5], high=[-0.5, 1, 1], size=(250, 3))
    group_2_labels = np.ones(250)

    # Group 3: Negative x, negative y, positive z corner
    group_3_features = np.random.uniform(low=[-1, -1, 0.5], high=[-0.5, -0.5, 1], size=(250, 3))
    group_3_labels = np.zeros(250)

    # Group 4: Positive x, negative y, positive z corner
    group_4_features = np.random.uniform(low=[0.5, -1, 0.5], high=[1, -0.5, 1], size=(250, 3))
    group_4_labels = np.ones(250)

    # Combine all groups
    dataset_a_features = np.concatenate((group_1_features, group_2_features, group_3_features, group_4_features), axis=0)
    dataset_a_labels = np.concatenate((group_1_labels, group_2_labels, group_3_labels, group_4_labels), axis=0)

    # Dataset B - Linearly Seperable
    mean_b_0 = [0, 0, 0]
    cov_b_0 = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    dataset_b_0 = multivariate_normal(mean_b_0, cov_b_0)
    dataset_b_features_0 = dataset_b_0.rvs(size=500)
    dataset_b_labels_0 = np.zeros(500)

    mean_b_1 = [10, 10, 10]
    cov_b_1 = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    dataset_b_1 = multivariate_normal(mean_b_1, cov_b_1)
    dataset_b_features_1 = dataset_b_1.rvs(size=500)
    dataset_b_labels_1 = np.ones(500)

    # Make features linearly dependent
    dataset_b_features_0[:, 1] = 2 * dataset_b_features_0[:, 0] + 3 * dataset_b_features_0[:, 2]
    dataset_b_features_1[:, 1] = 2 * dataset_b_features_1[:, 0] + 3 * dataset_b_features_1[:, 2]

    dataset_b_features = np.concatenate((dataset_b_features_0, dataset_b_features_1), axis=0)
    dataset_b_labels = np.concatenate((dataset_b_labels_0, dataset_b_labels_1), axis=0)
    
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }