import numpy as np 
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy import io

# np.random.seed(12)
# Load Data
mat = io.loadmat('../MNIST_database.mat')
x_train = mat['train_data'].T           # shape (1000,784)    
y_train = mat['train_classlabel'].T     # shape (1000,1)
x_test = mat['test_data'].T             # shape (250,784)
y_test = mat['test_classlabel'].T       # shape (250,1)

# re_mark labels accordingly my matrix No. A0117981X, label digits 8 and 1 as 1
x_train = x_train[np.where((y_train !=8) & (y_train !=1))[0]]

def init_grid(h,w):
    k = 0
    grid = np.zeros((h*w,2))
    for i in range(h):
        for j in range(w):
            grid[k,:] = [i,j]
            k += 1
    return grid

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

#conceptual
plt.figure()
for i in range(100):
    index = winnerIndex(updated_w[i],x_train)
    plt.subplot(10,10,i+1)  
    plt.imshow(x_train[index].reshape(28,28).T,cmap='gray')
    plt.axis('off')
plt.show()

##Visualization of Weight
plt.figure()
for j in range(100):
    
    plt.subplot(10,10,j+1)  
    plt.imshow(updated_w[j].reshape(28,28).T,cmap='gray')
    plt.axis('off')
plt.show()
