import numpy as np 
import pandas as pd
from scipy.optimize import fmin_tnc

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    print(np.dot(x.T, probability(theta, x)-y))
    return (1 / m) * np.dot(x.T, probability(theta,x) - y)

def update_theta(theta, x, y, eta):
    grad = gradient(theta, x, y)
    grad = eta * grad
    theta = theta - grad
    return theta

def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]

def fit2(x, y, theta, eta, iters):
    cost = []

    for i in range(iters):
        theta = update_theta(theta, x, y, eta)
        cost_iter = cost_function(theta, x, y)
        cost.append(cost_iter)

        if i%1000 == 0:
            print("iter: %s cost: %s\n", str(i), str(cost))
    
    return theta, cost

def predict(x):
    theta = parameters[:, np.newaxis]
    return probability(theta, x)

def accuracy(x, actual_classes, probab_threshold=0.5):
    predicted_classes = (predict(x) >= 
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100

# df = pd.read_csv("../../data/marks.data", header=None)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# X = np.c_[np.ones((X.shape[0], 1)), X]
# y = y[:, np.newaxis]
# theta = np.zeros((X.shape[1], 1))

df = pd.read_csv("../../data/breast-cancer-wisconsin.data", header=None)
X = df.iloc[:, 1:-1]
X = X.replace('?', 1)

y = df.iloc[:, -1]
y = y.replace(2, 0) # benign is 2 in dataset, so represent it with 0
y = y.replace(4, 1) # malignant is 4 in dataset, so represent with 1

X = np.c_[np.ones((X.shape[0], 1)), X]
X = X.astype(float)
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

print(X.shape)
print(y.shape)
print(theta.shape)

parameters = fit(X, y, theta)
print(parameters)

print(accuracy(X, y.flatten()))

#fit2(X, y, theta, 0.0001, 5000)
#print(probability(theta, X))
print(gradient(theta, X, y))