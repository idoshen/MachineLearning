import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    sum = 0

    def gini_step(count, datasetSize):
        return (count / datasetSize)**2
        

    _, countOfLabels = np.unique(data[:, -1], return_counts=True)
    datasetSize = data.shape[0]

    gini = 1 - np.sum(np.vectorize(gini_step)(countOfLabels, datasetSize))

    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    sum = 0

    def entropy_step(count, datasetSize):
        return (count / datasetSize) * np.log(count / datasetSize)

    _, countOfLabels = np.unique(data[:, -1], return_counts=True)
    datasetSize = data.shape[0]
    
    entropy = - np.sum(np.vectorize(entropy_step)(countOfLabels, datasetSize))

    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} 

    featureValues = np.unique(data[:, feature])

    totalDataImpurity = impurity_func(data)
    splitImpurity = 0
    splitInfo = 0

    for value in featureValues:
        split = data[data[:, feature] == value]
        groups[value] = split
        impurity = impurity_func(split)
        proportion = split.shape[0] / data.shape[0]
        splitImpurity += proportion * impurity
        splitInfo -= proportion * np.log(proportion)

    goodness = totalDataImpurity - splitImpurity

    if gain_ratio:
        if splitInfo == 0:
            goodness = 0
        else:
            goodness /= splitInfo

    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        max = 0
        labels, counts = np.unique(self.data[:, self.feature], return_counts=True)
        dict = {labels[i]: counts[i] for i in range(len(labels))}

        for label in labels:
            if (dict[label] > max):
                pred = label
                max = dict[label]

        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        if (self.depth >= self.max_depth or impurity_func(self.data) <= 0.0): 
            self.terminal = True
            return

        bestFeature = -1
        maxSplit = 0
        bestgroups = {}
        for feature in range(self.data.shape[1] - 1):
            goodness, groups = goodness_of_split(self.data, feature, impurity_func, gain_ratio=self.gain_ratio)
            if (goodness > maxSplit):
                maxSplit = goodness
                bestFeature = feature
                bestgroups = groups

        self.feature = bestFeature

        chi_from_table = -1
        chi_from_test = 0

        degreeOfFreedom = len(bestgroups.items()) - 1
        if self.chi != 1:
            if degreeOfFreedom <= 0:
                return
            else:
                chi_from_table = chi_table[degreeOfFreedom][self.chi]
                chi_from_test = chi_test(self, bestgroups.keys())

        
        if  chi_from_table < chi_from_test:
            for value, group in bestgroups.items():
                newNode = DecisionNode(group, -1,self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(newNode, value)
                newNode.split(impurity_func)


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = DecisionNode(data, feature=-1, depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    root.split(impurity)
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None

    while not root.terminal:
        currentFeature = root.feature
        answer = instance[currentFeature]
        if answer not in root.children_values:
            break
        indexOfChildChosen = root.children_values.index(answer)
        root = root.children[indexOfChildChosen]
    
    pred = root.pred

    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    counterOfCorrectPredictions = 0
   
    for instance in dataset:
         if predict(node, instance) == instance[-1]:
             counterOfCorrectPredictions += 1

    accuracy = counterOfCorrectPredictions / dataset.shape[0]

    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        root = build_tree(X_train, calc_entropy, gain_ratio=True, chi=1, max_depth=max_depth)
        accuracyTrain = calc_accuracy(root, X_train)
        accuracyTest = calc_accuracy(root, X_test)
        training.append(accuracyTrain)
        testing.append(accuracyTest)

    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    for p in p_values:
        root = build_tree(X_train, calc_entropy, gain_ratio=True, chi=p, max_depth=1000)
        accuracyTrain = calc_accuracy(root, X_train)
        accuracyTest = calc_accuracy(root, X_test)
        chi_training_acc.append(accuracyTrain)
        chi_testing_acc.append(accuracyTest)
        depth.append(tree_depth(root))
    
    return chi_training_acc, chi_testing_acc, depth

def tree_depth(root):
    if root.terminal:
        return 0
    else:
        child_depth = [tree_depth(child) for child in root.children]
        if len(child_depth) == 0:
            return 1
        else:
            return max(child_depth) + 1
    

def chi_test(node, values):

    chi_square_statistic = 0

    labels, counts = np.unique(node.data[:, node.feature], return_counts=True)
    
    Data = node.data[:, [node.feature, -1]]
    prob_e = np.where((Data[:, 1] == 'e'))[0].shape[0] / node.data.shape[0]
    prob_p = np.where((Data[:, 1] == 'p'))[0].shape[0] / node.data.shape[0]

    for value in values:
        df = counts[np.where(labels == value)][0]
        pf = np.where((Data[:, 0] == value) & (Data[:, 1] == 'e'))[0].shape[0]
        nf = np.where((Data[:, 0] == value) & (Data[:, 1] == 'p'))[0].shape[0]
        E_e = df * prob_e
        E_p = df * prob_p
        chi_square_statistic += (((pf - E_e)**2) /E_e) + (((nf - E_p)**2) /E_p)

    return chi_square_statistic


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = 0
    
    stack = [node]

    while stack:
        currentNode = stack.pop()
        n_nodes += 1
        if currentNode.children:
            for child in currentNode.children:
                stack.append(child)
        
    return n_nodes






