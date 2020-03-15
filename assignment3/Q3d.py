import numpy as np 
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy import io


# Load Data
mat = io.loadmat('../MNIST_database.mat')
x_train = mat['train_data'].T           # shape (1000,784)    
y_train = mat['train_classlabel'].T     # shape (1000,1)
x_test = mat['test_data'].T             # shape (250,784)
y_test = mat['test_classlabel'].T       # shape (250,1)

# re_mark labels accordingly my matrix No. A0117981X, label digits 8 and 1 as 1
idx_train = np.where((y_train != 8) & (y_train != 1))[0]
idx_test = np.where((y_test !=8) & (y_test !=1))[0]
y_train = y_train[idx_train]
x_train = x_train[idx_train]

x_test = x_test[idx_test]
y_test = y_test[idx_test]

x_test = x_test[np.where((y_test !=8) & (y_test !=1))[0]]

def init_grid(h,w):
    k = 0
    grid = np.zeros((h*w,2))
    for i in range(h):
        for j in range(w):
            grid[k,:] = [i,j]
            k += 1
    return grid

np.random.seed(6)
def init_weights(h,w,n):  # m-number of neurons, n-dimension of feature
    weights = np.random.rand(h*w,n)
    return weights

# find min dist and return winer
def winnerIndex(x,y):
    a = np.array(x).reshape(1,y.shape[1])
    d = cdist(a,y,metric='euclidean')
    idx = np.argmin(d)
    return idx

def adjustWeight(w,x,iter):
    time_ew = iter /np.log(sigma_init)    
    for i in range(iteration):
        for j in range(x.shape[0]):
            # winner index
            idx = winnerIndex(x[j],w)
            a = np.array(grid[idx]).reshape(1,grid.shape[1])
            d = cdist(a,grid,metric='euclidean')

            # update learning rate
            LR = learn_rate*np.exp(-(i+1)/iter)
            # update effective width
            sigma = sigma_init*np.exp(-(i+1)/time_ew)
           
            # time varying neibourhood func            
            h1 = np.exp(-(d**2)/(2*sigma**2)).reshape(100,1)
            alpha = LR*h1            
            # update weights
            w = (1-alpha)*w + alpha*x[j]
    return w

w = 10
h = 10
m = w*h
n = 28*28
learn_rate = 0.1
iteration = 10
sigma_init = np.sqrt(w**2+h**2)/2
grid = init_grid(h,w)
weights = init_weights(h,w,n)
updated_w = adjustWeight(weights,x_train,iteration)


W_labels = []
for i in range(100):
    index = winnerIndex(updated_w[i],x_train)
    W_labels.append(y_train[index][0])

test_Labels =[]
for j in range(x_test.shape[0]):
    index = winnerIndex(x_test[j],updated_w)
    test_Labels.append(W_labels[index])
     
error =0
for k in range(x_test.shape[0]):
    if test_Labels[k] != y_test[k][0]:
        error +=1
# print(error)
acc = 1-error/x_test.shape[0]
print('The classification accuracy of testing set is %.2f' %(acc*100) +'%')