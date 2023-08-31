from calendar import day_abbr
from operator import indexOf
import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.3,
            (1, 1): 0.4
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.08,
            (0, 0, 1): 0.02,
            (0, 1, 0): 0.12,
            (0, 1, 1): 0.08,
            (1, 0, 0): 0.12,
            (1, 0, 1): 0.08,
            (1, 1, 0): 0.18,
            (1, 1, 1): 0.32,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
       
        array = []

        for y in range(2):
            for x in range(2):
                array.append(np.isclose(X[x], X_Y[(x,y)]/Y[y]))

        return not np.all(array)

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
       
        array = []

        for y in range(2):
            for x in range(2):
                for c in range(2):
                    P_XYC = X_Y_C[(x,y,c)] / C[c]
                    P_XC = X_C[(x,c)] / C[c]
                    P_YC = Y_C[(y,c)] / C[c]
                    array.append(np.isclose(P_XC*P_YC, P_XYC))

        return  np.all(array)


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    numerator = (rate**k)*(np.e**(-rate))
    denominator = np.math.factorial(k)
    log_p = np.log(numerator/denominator)
    
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []
    sum = 0

    for rate in rates:
        for sample in samples:
            sum += poisson_log_pmf(sample, rate)
        likelihoods.append(sum)
        sum = 0

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    
    max = np.max(likelihoods)
    index = likelihoods.index(max)
    rate = rates[index]
    
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    n = samples.shape[0]
    sum = 0

    for sample in samples:
        sum += sample
    mean = sum / n

    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None

    numerator = np.exp(-0.5 * ((x - mean) / std)**2)
    denominator = std * np.sqrt(2 * np.pi)
    p = numerator / denominator
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """

        mean = []
        std = []
        data = dataset[dataset[:, -1] == class_value]
        n = data.shape[0]

        for col in range(data.shape[1] - 1):
            column_data = data[:, col]
            sum_mean = np.sum(column_data)
            mean.append(sum_mean / n)
            sum_std = np.sum((column_data - mean[col])**2)
            std.append(np.sqrt(sum_std / n))

        self.mean = mean
        self.std = std
        self.data = data
        self.classVal = class_value

        self.prior = n / dataset.shape[0]

    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """

        return self.prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """

        likelihood = normal_pdf(x[0], self.mean[0], self.std[0]) * normal_pdf(x[1], self.mean[1], self.std[1])

        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()

        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0
       
        posterior_zero = self.ccd0.get_instance_posterior(x)
        posterior_one = self.ccd1.get_instance_posterior(x)
        if (posterior_zero <= posterior_one):
            pred = 1
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    test_set_size = test_set.shape[0]
    counter = 0

    for instance in test_set:
        prediction = map_classifier.predict(instance)
        classification = instance[-1]
        if prediction == classification:
            counter += 1

    acc = counter / test_set_size

    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None

    # print("x =", x)
    # print("mean =", mean)
    # print("cov =", cov)
    # print("cov shape = ", cov.shape)
    e_power = -0.5 * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), (x - mean))
    numerator = (np.linalg.det(cov)**(-0.5)) * np.exp(e_power)
    pdf = numerator / (2*np.pi)

    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """

        mean = []
        data = dataset[dataset[:, -1] == class_value]
        n = data.shape[0]

        for col in range(data.shape[1] - 1):
            column_data = data[:, col]
            sum_mean = np.sum(column_data)
            mean.append(sum_mean / n)
            
        self.cov = np.cov(data[:, :-1], rowvar=False)
        self.mean = mean
        self.data = data
        self.classVal = class_value

        self.prior = data.shape[0]/ dataset.shape[0]
        
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = self.prior
        
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x[: -1], self.mean, self.cov)
        
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0
       
        prior_zero = self.ccd0.get_prior()
        prior_one = self.ccd1.get_prior()
        if (prior_zero <= prior_one):
            pred = 1

        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0
       
        likelihood_zero = self.ccd0.get_instance_likelihood(x)
        likelihood_one = self.ccd1.get_instance_likelihood(x)
        if (likelihood_zero <=  likelihood_one):
            pred = 1

        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        data = dataset[dataset[:, -1] == class_value]
        self.data = data
        self.classVal = class_value
        self.prior = data.shape[0]/ dataset.shape[0]
        
        uniqueAt = []

        for attribute in range(dataset.shape[1] - 1):
            col = dataset[:, attribute]
            currentUnique = np.unique(col)
            uniqueAt.append(currentUnique)

        self.uniqueAt = uniqueAt
        # print(uniqueAt)

    def get_prior(self):
        """
        Returns the prior probability of the class 
        according to the dataset distribution.
        """
        prior = self.prior

        return prior

    def laplace_Probability(self, n_ij, n_i, v_j):
        return ((n_ij + 1) / (n_i + v_j))
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1
        n_i = self.data.shape[0]
        
        for attribute in range(x.shape[0] - 1):
            if x[attribute] in self.uniqueAt[attribute]:
                n_ij = np.count_nonzero(self.data[:, attribute] == x[attribute])
                v_j = len(np.unique(self.data[:, attribute]))
                likelihood *= self.laplace_Probability(n_ij, n_i, v_j)
            else:
                likelihood *= EPSILLON
            
        
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 0
       
        posterior_zero = self.ccd0.get_instance_posterior(x)
        posterior_one = self.ccd1.get_instance_posterior(x)
        if (posterior_zero <= posterior_one):
            pred = 1
        
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        test_set_size = test_set.shape[0]
        counter = 0

        for instance in test_set:
            prediction = self.predict(instance)
            classification = instance[-1]
            if prediction == classification:
                counter += 1

        acc = counter / test_set_size
        return acc


