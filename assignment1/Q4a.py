import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,0],[1,0.8],[1,1.6],[1,3],[1,4],[1,5]])
d = np.array([[0.5] , [1], [4], [5], [6], [9]])

w = np.mat(np.linalg.inv((np.mat(np.transpose(x)))*(np.mat(x))))*np.mat(np.transpose(x))*np.mat(d)
# w0 = 0.38733906 w1 = 1.60944206
X = np.arange(0,6,1)
y = w[1,0]*X + w[0,0]

plt.figure()
plt.title('Standard Linear Least-Squares (LLS)')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X,y,color= 'red',label = "LLS")
plt.plot(x[:,1:2],d[:,:],'bx')
plt.text(2, 3, "y = "+str("%.2f" %w[1,0]) +"x +"+str("%.2f" %w[0,0]), size = 15,\
         family = "fantasy", color = "b", style = "italic", weight = "light",\
         bbox = dict(facecolor = "r", alpha = 0.2))

plt.legend()
plt.show()



