import numpy as np
import math
import pandas as pd


# Predict with a certain input and a group of theta
def model_func(x, theta):
    y_hat = []
    for obs in x:
        z = 0
        for i in range(len(obs)):
            z = z + obs[i] * theta[i]
        y_obs = 1 / (1 + np.exp(-z))
        y_hat.append(y_obs)
    return np.array(y_hat)


# Cost of an output
def cost(y, y_hat):
    y_hat = y_hat.tolist()
    return -y * [math.log(i) for i in y_hat] - (1 - y) * [math.log(1 - i) for i in y_hat]


# Loss function
def loss(y, y_hat):
    return cost(y, y_hat).mean()


# Gradient at a certain input of the loss function
def gradient(x, theta, y):
    y_hat = model_func(x, theta)
    error = y_hat - y
    return -(1.0/len(x)) * error.dot(x)


# Train the model by gradient decent method
def gradient_decent(x, theta, y, alpha, lamda, stop=.01):
    grad = gradient(x, theta, y)
    my_loss = 0
    while np.linalg.norm(grad) > stop :
        theta = theta * regularization(theta, lamda) + grad * alpha
        my_loss = loss(y, model_func(x, theta))
        grad = gradient(x, theta, y)
        # print(my_loss)
    return theta, my_loss


def regularization(theta, lamda):
    return 1.0 - lamda * (1.0 / len(theta)) * theta


# Evaluate the accuracy of the validation result to report the performance of the trained model
def accuracy(y, y_hat):
    res = []
    TP = 0
    TN = 0
    P = 0
    N = 0
    for i in range(len(y_hat)):
        prediction = y_hat[i]
        obs = y[i]
        if prediction >= 0.5:
            res.append(1)
            P += 1
            if obs == 1:
                TP += 1
        else:
            res.append(0)
            N += 1
            if obs == 0:
                TN += 1
    accu = (TN + TP) / (P + N)
    return res, accu

# Classify the test data and export the result
def predict(x, theta):
    prediction = model_func(x, theta)
    res = []
    res_id = []
    for p in prediction:
        if p >= 0.5:
            res_id.append(1)
            res.append('plastic case')
        else:
            res_id.append(0)
            res.append('book')
    res = pd.DataFrame({'class ID': res_id, 'class': res})
    res.to_csv('PredictedClasses.csv')
    return res

