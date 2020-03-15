import numpy as np 
import matplotlib.pyplot as plt

x_train = np.arange(-1,1,0.05)
y_train = 1.2*np.sin(np.pi*x_train) - np.cos(2.4*np.pi*x_train)

x_test = np.arange(-1,1,0.01)
y_test = 1.2*np.sin(np.pi*x_test) - np.cos(2.4*np.pi*x_test)

np.random.seed(2)
g_noise = np.random.normal(0,1,y_train.shape[0])
y_train_n = y_train + 0.3*g_noise

sigma = 0.1
lmd_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]

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

def weights(x,d,lmd):
    phi = phi_matrix(x)
    I = np.identity(phi.shape[0])
    return np.linalg.inv(np.dot(phi.T,phi) + lmd*I).dot(phi.T).dot(d)

def predict(x_train,x_test,weights):
    y_pre = []
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(x_train.shape[0]):
            r = x_test[j] - x_train[i]
            y += weights[i]*guassian(r)
        y_pre.append(y) 
    return y_pre

pre_test = []
for i in range(len(lmd_set)):
    w = weights(x_train,y_train_n,lmd_set[i])
    pre_test.append(predict(x_train,x_test,w))


plt.figure()
for i in list(range(0,9,2)):
    plt.subplot(1,2,1)
    plt.plot(x_test,y_test,color = 'red',label ='Test Output')
    plt.plot(x_test,pre_test[i],color = 'blue',label ='Predicted Output')
    plt.title('The result of the exact interpolation with regularization λ = '+ str(lmd_set[i]))
    plt.ylim(-3,3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_test,y_test,color = 'red',label ='Test Output')
    plt.plot(x_test,pre_test[i+1],color = 'blue',label ='Predicted Output')
    plt.title('The result of the exact interpolation with regularization λ = '+ str(lmd_set[i+1]))
    plt.ylim(-3,3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    plt.clf()
    
err_set = []
for i in range(len(lmd_set)):
    err_set.append(np.mean(abs((pre_test[i] - y_test)/y_test)))
print(min(err_set))
plt.plot(lmd_set,err_set)
plt.xlabel('Regularization Factor')
plt.ylabel('Average Relative Error')
plt.title('Average Relative Error versus the Regularization Factor on Test Data Set')
plt.show()