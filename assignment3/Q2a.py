import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# Load Data
mat = io.loadmat('../MNIST_database.mat')
x_train_raw = mat['train_data'].T           # shape (1000,784)    
y_train_raw = mat['train_classlabel'].T     # shape (1000,1)
x_test_raw = mat['test_data'].T             # shape (250,784)
y_test_raw = mat['test_classlabel'].T       # shape (250,1)
# my matrix No. A0117981X, mark digit 8 class 1 and digit 1 as class 0
x_train = x_train_raw[np.where((y_train_raw ==8) | (y_train_raw ==1))[0]]
y_train = y_train_raw[np.where((y_train_raw ==8) | (y_train_raw ==1))[0]]
x_test = x_test_raw[np.where((y_test_raw ==8) | (y_test_raw ==1))[0]]
y_test = y_test_raw[np.where((y_test_raw ==8) | (y_test_raw ==1))[0]]
y_train = np.where(y_train == 8, 1,0)
y_test = np.where(y_test == 8, 1,0)
np.random.seed(8)

def guassian(r):
    return np.exp(-r**2/(2*sigma**2))

def phi_matrix(x):
    d = x.shape[0]
    phi = np.zeros((d,d))
    for i in range(d):
        for j in range(d):            
            r = x[i]-x[j]
            phi[i][j] = guassian(np.linalg.norm(r))
    return phi

def weights(x,d):
    phi = phi_matrix(x)
    return np.linalg.inv(phi).dot(d)

def predict(x_train,x_test,W):
    y_pre = np.zeros((x_test.shape[0], 1))
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(x_train.shape[0]):
            r = x_test[j] - x_train[i]
            y += W[i]*guassian(np.linalg.norm(r))
        y_pre[j][0] = y    
    return y_pre

sigma = 100
w = weights(x_train,y_train)
pre_test = predict(x_train,x_test,w)
pre_train = predict(x_train,x_train,w)

TrAcc = []
TeAcc = []
TrN = y_train.shape[0]
TeN = y_test.shape[0]
Threshold = []
for i in range(1000):
    t = (np.amax(pre_train) - np.amin(pre_train))*i/1000 + np.amin(pre_train)
    Threshold.append(t)
    TrAcc.append((np.sum(y_train[pre_train<t]==0) + np.sum(y_train[pre_train>=t]==1)) / TrN)
    TeAcc.append((np.sum(y_test[pre_test<t]==0) + np.sum(y_test[pre_test>=t]==1)) / TeN)

plt.figure()
plt.plot(Threshold,TrAcc,Threshold,TeAcc)
plt.title('Accuracy in Exact Interpolation Method without regularization')
plt.legend(('Train ', 'Test'))
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
