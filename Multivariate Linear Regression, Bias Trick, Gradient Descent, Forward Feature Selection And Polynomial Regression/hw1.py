# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    meanX = np.mean(X , axis = 0)
    meanY = np.mean(y , axis = 0)
    maxX = np.max(X , axis = 0)
    maxY = np.max(y , axis = 0)
    minX = np.min(X , axis = 0)
    minY = np.min(y , axis = 0)
    X = (X - meanX)/(maxX - minX)
    y = (y - meanY)/(maxY - minY)
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    if (X.ndim == 1):
        X = np.reshape(X, (-1, 1))

    a = np.ones((X.shape[0],1))

    X = np.concatenate((a, X), axis=1)
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    J = np.sum((np.dot(X, theta) - y)**2)/(2*X.shape[0])
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    
    sum = 0

    for i in range(num_iters):
        sum = np.dot((np.dot(X, theta) - y),X)
        J = np.multiply(alpha, np.divide(sum, X.shape[0]))
        theta = np.subtract(theta, J)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    X_inv = np.linalg.inv(np.dot(X.T, X))
    pinv = np.dot(X_inv, X.T)
    pinv_theta = np.dot(pinv, y)

    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration


    for i in range(num_iters):
        sum = np.dot((np.dot(X, theta) - y),X)
        J = np.multiply(alpha, np.divide(sum, X.shape[0]))
        theta = np.subtract(theta, J)
        J_history.append(compute_cost(X, y, theta))
        if (i > 0) and (J_history[-2] - J_history[-1] < 1e-8): break 


    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    theta = np.random.random(size=X_train.shape[1])
    
    for a in alphas:
        x = efficient_gradient_descent(X_train, y_train, theta, a, iterations)
        alpha_dict[a] = compute_cost(X_val, y_val, x[0])

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    
    X_train_bias = apply_bias_trick(X_train)
    X_val_bias = apply_bias_trick(X_val)

    selected_features.append(0)

    random_theta = np.zeros(6)
    
    while len(selected_features) <= 5:
        index = -1
        min_cost = 1
        for i in range(1, X_train_bias.shape[1]):
            
            if i not in selected_features:
                
                selected_features.append(i)
                theta, _ = efficient_gradient_descent(X_train_bias[:,selected_features], y_train, random_theta[:len(selected_features)], best_alpha, iterations)
                # theta = compute_pinv(X_train_bias[:,selected_features], y_train)
                cost = compute_cost(X_val_bias[:,selected_features], y_val, theta)

                if cost < min_cost:
                    min_cost = cost
                    index = i
                
                selected_features.pop()
            
        selected_features.append(index)

    selected_features.remove(0)

    selected_features = np.array(selected_features) - 1


    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    for feature1 in df.columns.tolist():
        result = pd.DataFrame()
        for feature2 in df.columns.tolist():
            result[feature1 + '*' + feature2] = df[feature1] * df[feature2]
        df_poly = pd.concat((df_poly, result), axis=1)

            
        df = df.drop(feature1, axis=1)

    return df_poly