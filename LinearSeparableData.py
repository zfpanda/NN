from numpy import *
from Perceptron import Perceptron
import matplotlib.pyplot as plt 
def makeLinearSeparableData(n_samples,n_features):
    w = random.rand(n_features)*2 -1
    # numFeatures = len(weights)
    dataSet = zeros((n_samples,n_features+1))

    for i in range(n_samples):
        x = random.rand(1,n_features) * 20 - 10
        innerProduct = sum(w*x)
        if innerProduct <= 0:
            dataSet[i] = append(x,-1)
        else:
            dataSet[i] = append(x,1)

    return dataSet

# print(random.rand(1,2) * 20 - 10) #[[-5.6349041  -5.06855272]]
# print(random.rand(2)*2 -1)


def plotData(dataSet,classifier):
    ''' (array) -> figure

    Plot a figure of dataSet

    '''
    
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    x = range(-10,10) 
    y = (classifier[0] + classifier[1]*x)/classifier[2]*(-1)
    labels = array(dataSet[:,2])
    idx_1 = where(dataSet[:,2]==1)
    p1 = ax.scatter(dataSet[idx_1,0], dataSet[idx_1,1], marker='o', color='g', label=1, s=20)
    idx_2 = where(dataSet[:,2]==-1)
    p2 = ax.scatter(dataSet[idx_2,0], dataSet[idx_2,1], marker='x', color='r', label=2, s=20)
    plt.legend(loc = 'upper right')
     
    plt.plot(x,y)


    plt.show()

data = makeLinearSeparableData(100,2)


X = data[...,[0,1]]
y = data[...,[-1]]

ppn = Perceptron(eta = 0.1, n_iter = 50)
ppn.fit(X,y)
# print(ppn.fit(X,y).w)
f = plotData(data,ppn.fit(X,y).w)
# plt.plot(range(1,len(ppn.errors) + 1), ppn.errors, marker = 'o')
# plt.xlabel('Epoches')  
# plt.ylabel('Number of misclassification')
# plt.show()