# coding=utf-8

import numpy as np
import random
from numpy import genfromtxt
import pandas as pd

#
# def getData(data_set):
#     m, n = np.shape(data_set)
#     train = dataSet.iloc[:, [0, -1]]
#     params = pd.DataFrame(np.ones((m, 1)))
#     train_data = pd.concat([train, params], axis=1)
#     train_label = dataSet.iloc[:, -1]
#     return train_data, train_label

def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,:-1] = dataSet[:,:-1]
    trainLabel = dataSet[:,-1]
    return trainData, trainLabel

def mse(Y, Y_hat):
    return np.mean(np.square(Y - Y_hat), axis=0)


# def gd(X, theta, alpha):
#     Y_hat = X.dot(theta)
#     loss = mse(Y_hat, Y)
#     prevtheta = theta.copy()
#     thetagrad = np.mean(X * ((Y_hat - Y).reshape(-1, 1)), axis=0)
#     theta -= thetagrad * alpha
#     return prevtheta, loss


def batchGradientDescent(x, y, theta, alpha, epoch):
    for i in range(0, epoch):
        pre = np.dot(x, theta)
        print('loss:{}'.format(np.mean(pre - y)))
        gradient = np.dot(pre - y, x)
        theta = theta - alpha * gradient
    return theta


def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n + 1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP


#dataPath = "/Users/tengyujia/Desktop/house.csv"
#dataSet = pd.read_csv(dataPath, header=None)
dataPath = r"/Users/tengyujia/Desktop/house.csv"
dataSet = genfromtxt(dataPath, delimiter=',')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.00000001
epoch = 50
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, epoch)
x = np.array([[2104, 3], [1600, 3], [2400, 3], [1416, 2], [3000, 4]])
print(predict(x, theta))
