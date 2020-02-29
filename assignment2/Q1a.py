import numpy as np
import random
import matplotlib.pyplot as plt

def func_valley(xy):
    x = xy[0][0]
    y = xy[0][1]
    return (1-x)**2 + 100*(y-x**2)**2

def func_dx(x,y):
    return -2 + 2*x - 400*(y-x**2)*x

def func_dy(x,y):
    return 200*(y-x**2)

def grad_func(xy):
    x = xy[0][0]
    y = xy[0][1]
    return np.array([func_dx(x,y),func_dy(x,y)]).reshape(1,2)
    
xy = np.array([random.uniform(0,0.5),random.uniform(0,0.5)]).reshape(1,2)
# xy = np.array([0.08381745243420341,0.04145318944720866]).reshape(1,2)
print(xy)
iteration = 0
learning_rate = 0.001
x_trajectory = []
y_trajectory = []
v_xy = []
while func_valley(xy) > 0.0000001:
    x_trajectory.append(xy[0][0])
    y_trajectory.append(xy[0][1])
    v_xy.append(func_valley(xy)) 
    xy = xy - learning_rate*grad_func(xy)
    iteration += 1
print(iteration)
print('initial x value is '+ str(x_trajectory[0]) +' initial y value is '+ str(y_trajectory[0]))
print('final x value is '+ str(x_trajectory[-1]) +' final y value is '+ str(y_trajectory[-1]))
    
fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Trajectory of (x, y) after %d iterations' % iteration)
plt.xlabel ("x")
plt.ylabel ("y")
plt.plot(x_trajectory,y_trajectory)

plt.subplot(1,2,2)
plt.title('Value of f(x, y) versus number of iterations')
plt.xlabel ("Iteration")
plt.ylabel ("Value")
plt.plot(np.arange(iteration),v_xy)
fig.set_size_inches(8, 5, forward=True)
fig.tight_layout()
plt.savefig('Q1a.png')
plt.show()