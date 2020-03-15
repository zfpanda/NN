import numpy as np 
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

np.random.seed(6)
def initOutputLayer(m,n):  # m-number of neurons, n-dimension of feature
    layer = np.random.rand(m,n)
    return layer

# find min dist and return winer
def winnerIndex(x,y):
    a = np.array(x).reshape(1,y.shape[1])
    d = cdist(a,y,metric='euclidean')
    idx = np.argmin(d)
    return idx

def adjustWeight(layer,x,iter):
    time_ew = iter /np.log(sigma_init)
    w = layer
    for i in range(iteration):
        for j in range(x.shape[0]):
            #winner index
            idx = winnerIndex(x[j],w)
            # update learning rate
            LR = learn_rate*np.exp(-(i+1)/iter)
            # update effective width
            sigma = sigma_init*np.exp(-(i+1)/time_ew)
           
            # time varying neibourhood func
            N_index = np.arange(m).reshape(m,1)
            h = np.exp(-(N_index - idx)**2/(2*sigma**2))
            alpha = LR*h             
            # update weights
            w = (1-alpha)*w + alpha*x[j]
    return w


m = 25
n = 2
learn_rate = 0.1
iteration = 1000
sigma_init = 1 * m
t = np.linspace(-np.pi,np.pi,200)

trainX = np.array((t*np.sin(np.pi*np.sin(t)/t),1-abs(t)*np.cos(np.pi*np.sin(t)/t))).T

weights = adjustWeight(initOutputLayer(25,2),trainX,iteration)


plt.plot(trainX[:,0],trainX[:,1],)
plt.plot(weights[:,0],weights[:,1])
plt.title('Performance of SOM')
plt.legend(('Samples', 'Weights'))  
plt.show()
