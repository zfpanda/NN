import numpy as np 
import matplotlib.pyplot as plt
import random

x_train = np.arange(-1,1,0.05)
y_train = 1.2*np.sin(np.pi*x_train) - np.cos(2.4*np.pi*x_train)

x_test = np.arange(-1,1,0.01)
y_test = 1.2*np.sin(np.pi*x_test) - np.cos(2.4*np.pi*x_test)

np.random.seed(6)
g_noise = np.random.normal(0,1,y_train.shape[0])
y_train_n = y_train + 0.3*g_noise

x_centers = []
y_centers = []
M = 15
# Select random 15 data points
idx = random.sample(range(0,len(x_train)),M)

for i in range(len(idx)):
    x_centers.append(x_train[idx[i]])
    # y_centers.append(y_train_n[idx[i]])

d_max = max(x_centers) - min(x_centers)
sigma = d_max/np.sqrt(2*M)

def guassian(r):
    return np.exp(-r**2/(2*sigma**2))

def phi_matrix(x,mu):
    N = x.shape[0]
    M = len(mu)
    phi = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            r = x[i]-mu[j]
            phi[i][j] = guassian(r)
    return phi

def weights(mu,x,d):
    phi = phi_matrix(x,mu)
    return np.linalg.inv(np.dot(phi.T,phi)).dot(phi.T).dot(d)

def predict(x_train,x_test,weights):
    y_pre = []
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(len(x_train)):
            r = x_test[j] - x_train[i]
            y += weights[i]*guassian(r)
        y_pre.append(y) 
    return y_pre

w = weights(x_centers,x_train,y_train_n)
pre_test = predict(x_centers,x_test,w)
mean_error = np.mean(abs((pre_test - y_test)/y_test))
print(mean_error)

plt.figure()
plt.plot(x_test,y_test,x_test,pre_test)
plt.legend(('Test Output', 'Predicted Output'))
plt.title('result of fixed centres selected at random method')
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()