import numpy as np 
import matplotlib.pyplot as plt 
import random
from scipy import io
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Load Data
mat = io.loadmat('../MNIST_database.mat')
x_train_raw = mat['train_data'].T           # shape (1000,784)    
y_train_raw = mat['train_classlabel'].T     # shape (1000,1)
x_test_raw = mat['test_data'].T             # shape (250,784)
y_test_raw = mat['test_classlabel'].T       # shape (250,1)
# my matrix No. A0117981X, mark digit 8 class 1 and digit 1 as class 0
x_train = x_train_raw[np.where((y_train_raw ==8) | (y_train_raw ==1))[0]]
y_train = y_train_raw[np.where((y_train_raw ==8) | (y_train_raw ==1))[0]]
x_test = x_test_raw[np.where((y_test_raw ==8) | (y_test_raw ==1))[0]]
y_test = y_test_raw[np.where((y_test_raw ==8) | (y_test_raw ==1))[0]]
y_train = np.where(y_train == 8, 1,0)
y_test = np.where(y_test == 8, 1,0)
np.random.seed(8)

def euclDistance(vector1,vector2):
    return np.sqrt(np.sum(np.power(vector2 - vector1,2)))

def initCentroids(dataSet,k):
    nuSamples, dim = dataSet.shape  
    index = random.sample(range(nuSamples),k) #randomly generate k non repeated num, return in list
    centroids = np.array([dataSet[i,:] for i in index]) # create k centroids from the array
    # centroids = x_train[np.random.randint(1000, size=2), :] random centres
    return centroids

def k_means(dataSet,k):
    nuSamples = dataSet.shape[0]
    clusterAssment = np.zeros((nuSamples,2))
    centroids = initCentroids(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False 
        for i in range(nuSamples):
            minDistance = float("inf")
            ownGroup = 0
            for j in range(k):
                distance = euclDistance(centroids[j,:],dataSet[i,:])
                if distance < minDistance:
                    minDistance = distance
                    ownGroup = j
            if clusterAssment[i,1] != ownGroup:
                clusterAssment[i,1] = ownGroup
                clusterChanged = True
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,1]==j)[0]]           
            centroids[j,:] = np.mean(pointsInCluster,axis =0)
        return centroids,clusterAssment

k = 2
centroids, clusterAssment = k_means(x_train, k)

d = cdist(centroids,centroids,metric='euclidean')
d_max = np.amax(d)
sigma = d_max/np.sqrt(2*k)

def guassian(r):
    return np.exp(-r**2/(2*sigma**2))

def phi_matrix(x,centre):
    N = x.shape[0]
    M = centre.shape[0]
    phi = np.zeros((N,M))
    for i in range(N):
        for j in range(M):            
            r = x[i]-x[j]
            phi[i][j] = guassian(np.linalg.norm(r))
    return phi

def weights(x,d,centre):   
    phi = phi_matrix(x,centre)
    return np.linalg.inv(np.dot(phi.T,phi)).dot(phi.T).dot(d)

def predict(centre,x_test,W):
    y_pre = np.zeros((x_test.shape[0], 1))
    for j in range(x_test.shape[0]):
        y = 0
        for i in range(centre.shape[0]):
            r = x_test[j] - centre[i]
            y += W[i]*guassian(np.linalg.norm(r))
        y_pre[j][0] = y    
    return y_pre


w = weights(x_train,y_train,centroids)
pre_test = predict(centroids,x_test,w)
pre_train = predict(centroids,x_train,w)


TrAcc = []
TeAcc = []
thr = []
TrN = y_train.shape[0]
TeN = y_test.shape[0]
Thr = []
for i in range(1000):
    t = (np.amax(pre_train) - np.amin(pre_train))*i/1000 + np.amin(pre_train)
    Thr.append(t)

    TrAcc.append((np.sum(y_train[pre_train<t]==0) + np.sum(y_train[pre_train>=t]==1)) / TrN)
    TeAcc.append((np.sum(y_test[pre_test<t]==0) + np.sum(y_test[pre_test>=t]==1)) / TeN)

mean_value = np.zeros((2,784))
for i in range(k):
    mean_value[i,:] = np.mean(x_train[np.nonzero(y_train[:,0]==i)[0]],axis =0)

#降维
pca = PCA(2)
projected = pca.fit_transform(centroids)
classmean = pca.fit_transform(mean_value)
# class1 = pca.fit_transform(class1mean)
# #画出每个点的前两个主成份
# plt.scatter(projected[:,0], projected[:,1])
plt.scatter(projected[0:,0],projected[0:,1],s=200,c='red',marker='o',alpha=0.5,label='Centre 0')
plt.scatter(classmean[0:,0],classmean[0:,1],s=200,c='black',marker='o',alpha=0.5,label='mean 0')
# plt.annotate("(3,6)", xy = (3, 6), xytext = (4, 5), arrowprops = dict(facecolor = 'black', shrink = 0.1))
plt.scatter(projected[1:,0],projected[1:,1],s=200,c='blue',marker='o',alpha=0.5,label='Centre 1')

plt.scatter(classmean[1:,0],classmean[1:,1],s=200,c='green',marker='o',alpha=0.5,label='mean 1')

# plt.scatter(class1[0],class1[1],s=200,c='black',marker='o',alpha=0.5,label='C1ass 1')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()



# plt.figure()
# plt.plot(Thr,TrAcc,Thr,TeAcc)
# plt.title('Accuracy in Exact Interpolation Method regularization')
# plt.legend(('Tr ', 'Te'))
# plt.xlabel('Threshold')
# plt.ylabel('Accuracy')
plt.show()


