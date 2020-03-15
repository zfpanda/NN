import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# Load Data
mat = io.loadmat('../MNIST_database.mat')
x_train = mat['train_data'].T           # shape (1000,784)    
y_train = mat['train_classlabel'].T     # shape (1000,1)
x_test = mat['test_data'].T             # shape (250,784)
y_test = mat['test_classlabel'].T       # shape (250,1)

# mark labels accordingly my matrix No. A0117981X, label digits 8 and 1 as 1
y_train = np.where((y_train == 8) | (y_train == 1), 1,0)
y_test = np.where((y_test == 8) | (y_test == 1), 1,0)

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

lmd_set = [0,0.001,0.005]
lmd = 0
w = weights(x_train,y_train,lmd)
pre_test = predict(x_train,x_test,w)
pre_train = predict(x_train,x_train,w)

print(w.shape)

TrAcc = []
TeAcc = []
thr = []
TrN = y_train.shape[0]
TeN = y_test.shape[0]
Thr = []
for i in range(1000):
    t = (np.amax(pre_train) - np.amin(pre_train))*i/1000 + np.amin(pre_train)
    Thr.append(t)

    TrAcc.append((np.sum(y_train[pre_train<t]==0) + np.sum(y_train[pre_train>=t]==1)) / TrN)
    TeAcc.append((np.sum(y_test[pre_test<t]==0) + np.sum(y_test[pre_test>=t]==1)) / TeN)

plt.figure()
plt.plot(Thr,TrAcc,Thr,TeAcc)
plt.title('Performance of Exact Interpolation Method with regularization λ = ')
plt.legend(('Tr ', 'Te'))
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
