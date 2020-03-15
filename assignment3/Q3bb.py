import numpy as np 
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
np.random.seed(12)

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

def distEclud(x, weights):
    dist = []
    for w in weights:
        d = np.linalg.norm(x-w)
        dist.append(d)
    return np.array(dist)

def adjustWeight(w,x,iter):
    time_ew = iter /np.log(sigma_init)
    
    for i in range(iteration):
        for j in range(x.shape[0]):
            # winner index
            idx = winnerIndex(x[j],w)
            a = np.array(grid[idx]).reshape(1,grid.shape[1])
            d = cdist(a,grid,metric='euclidean')

            # index2 = (dist2 ; 1).nonzero()[0]
            # update learning rate
            LR = learn_rate*np.exp(-(i+1)/iter)
            # update effective width
            sigma = sigma_init*np.exp(-(i+1)/time_ew)
           
            # time varying neibourhood func
            h1 = np.exp(-(d**2)/(2*sigma**2)).reshape(25,1)
            alpha = LR*h1            
            # update weights
            w = (1-alpha)*w + alpha*x[j]
    return w

w = 5
h = 5
m = w*h
n = 2
learn_rate = 0.1
iteration = 100
sigma_init = np.sqrt(w**2+h**2)/2
grid = init_grid(h,w)
weights = init_weights(h,w,n)

trainX = -1 + 2*np.random.random((500,2))
w = adjustWeight(weights,trainX,iteration)


def plot_weights(weights, h=5, w=5, size=(6,6)):   
    x_axis = weights[:,0].reshape(h, w)
    y_axis = weights[:,1].reshape(h, w)
    plt.figure(figsize=size) 
    plt.title('Training Points and Weights') 
    plt.scatter(trainX[:,0],trainX[:,1],c='blue',label = 'Samples')  
    plt.scatter(weights[:,0],weights[:,1],marker = 'x',c='red',label = 'Weights')
    plt.legend(loc ='upper right')
    for i in range(h):
        plt.plot(x_axis[i], y_axis[i],c='red')
        plt.plot(x_axis.T[i], y_axis.T[i],c='red')   
    plt.show()

plot_weights(w,5,5,size=(6,6))


