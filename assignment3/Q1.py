import numpy as np 
import matplotlib.pyplot as plt

x_train = np.arange(-1,1,0.05)
y_train = 1.2*np.sin(np.pi*x_train) - np.cos(2.4*np.pi*x_train)

x_test = np.arange(-1,1,0.01)
y_test = 1.2*np.sin(np.pi*x_test) - np.cos(2.4*np.pi*x_test)

np.random.seed(6)
g_noise = np.random.normal(0,1,y_train.shape[0])
y_train_n = y_train + 0.3*g_noise

def guassian(r):
    return np.exp(-r**2/(2*sigma**2))

def phi_matrix(x):
    d = x.shape[0]
    phi = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            r = x[i]-x[j]
            phi[i][j] = guassian(r)
    return phi

def weights(x,d):
    phi = phi_matrix(x)
    return np.linalg.inv(phi).dot(d)

def predict(x_train,x_test,weights):
    y_pre = []
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(x_train.shape[0]):
            r = x_test[j] - x_train[i]
            y += weights[i]*guassian(r)
        y_pre.append(y) 
    return y_pre

sigma = 0.1

w = weights(x_train,y_train_n)
pre_test = predict(x_train,x_test,w)

mean_error = np.mean(abs((pre_test - y_test)/y_test))
print(mean_error)
plt.figure()
plt.plot(x_test,y_test,x_test,pre_test)
plt.legend(('Test Dataset', 'Predicted Dataset'))
plt.ylim(-3, 3)
plt.title('result of exact interpolation method')
plt.xlabel('x')
plt.ylabel('y')
plt.show()