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
sigma = 100
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

def weights(x,d,lmd):   
    phi = phi_matrix(x)
    I = np.identity(phi.shape[0])
    return np.linalg.inv(np.dot(phi.T,phi) + lmd*I).dot(phi.T).dot(d)

def predict(x_train,x_test,W):
    y_pre = np.zeros((x_test.shape[0], 1))
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(x_train.shape[0]):
            r = x_test[j] - x_train[i]
            y += W[i]*guassian(np.linalg.norm(r))
        y_pre[j][0] = y    
    return y_pre

lmd_set = [0,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]
pre_test = []
pre_train = []
for i in range(len(lmd_set)):
    w = weights(x_train,y_train,lmd_set[i])
    pre_train.append(predict(x_train,x_train,w))
    pre_test.append(predict(x_train,x_test,w)) 
    
TrN = y_train.shape[0]
TeN = y_test.shape[0]
TrAccuracy = []
TeAccuracy = []
thr = []
for j in range(len(lmd_set)):
    TrAcc = []
    TeAcc = []   
    Threshold = []    
    for i in range(1000):
        t = (np.amax(pre_train[j]) - np.amin(pre_train[j]))*i/1000 + np.amin(pre_train[j])
        Threshold.append(t)

        TrAcc.append((np.sum(y_train[pre_train[j]<t]==0) + np.sum(y_train[pre_train[j]>=t]==1)) / TrN)
        TeAcc.append((np.sum(y_test[pre_test[j]<t]==0) + np.sum(y_test[pre_test[j]>=t]==1)) / TeN)
    thr.append(Threshold)
    TrAccuracy.append(TrAcc)
    TeAccuracy.append(TeAcc)

plt.figure()
for i in list(range(0,9,2)):
    plt.subplot(1,2,1)
    plt.plot(thr[i],TrAccuracy[i],thr[i],TeAccuracy[i])
    plt.title('Performance with regularization λ = '+str(lmd_set[i]))
    plt.legend(('Train ', 'Test'))
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(thr[i+1],TrAccuracy[i+1],thr[i+1],TeAccuracy[i+1])
    plt.title('Performance with regularization λ = '+str(lmd_set[i+1]))
    plt.legend(('Train', 'Test'))
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.show()
    plt.clf()
