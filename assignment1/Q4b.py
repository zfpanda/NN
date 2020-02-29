import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,0],[1,0.8],[1,1.6],[1,3],[1,4],[1,5]])
d = np.array([[0.5] , [1], [4], [5], [6], [9]])

#learning rate
learn_rate = 0.009

# 100 epochs
iteration = 100

def train(x,iteration,learn_rate):
    #initial weights
    np.random.seed(0)
    weights = np.random.randn(1,x.shape[1])
    W = weights
    for i in range(iteration):
        for j in range(x.shape[0]):
            error = d[j] - np.mat(x[j,:])*np.transpose(weights)
            weights = weights + learn_rate*error*x[j,:]                
        W = np.append(W,weights,axis = 0)
        # print("Updated weights is " + str(weights))                  
    # print(str(iteration) + " times of epochs")
    return W
    

W_x = train(x,iteration,learn_rate)
# print(W_x)
X = np.arange(0,6,1)
y = W_x[W_x.shape[0]-1,1]*X + W_x[W_x.shape[0]-1,0]
plt.figure()
plt.subplot(1,2,1)
plt.title('LMS fitting result')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x[:,1:2],d,'ro')
plt.text(2, 3, "y = "+str("%.2f" %W_x[W_x.shape[0]-1,1]) +"x +"+str("%.2f" %W_x[W_x.shape[0]-1,0]), size = 15,\
         family = "fantasy", color = "b", style = "italic", weight = "light",\
         bbox = dict(facecolor = "r", alpha = 0.2))
plt.plot(X,y,color= 'green',label = "LMS")
plt.legend()

plt.subplot(1,2,2)
plt.title('trajectories of the weights VS learning steps')
plt.xlabel('learning steps')
plt.ylabel('weights')
plt.plot(range(W_x.shape[0]),W_x[:,0:1],'b^',range(W_x.shape[0]),W_x[:,1:2],'g*')
plt.legend(('W0','W1'),loc = 'upper right')
plt.show()