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
# np.random.seed(12)
def euclDistance(vector1,vector2):
    return np.sqrt(np.sum(np.power(vector2 - vector1,2)))

def initCentroids(dataSet,k):
    nuSamples, dim = dataSet.shape  
    index = random.sample(range(nuSamples),k) #randomly generate k non repeated num, return in list
    # centroids = np.array([dataSet[i,:] for i in index]) # create k centroids from the array
    centroids = x_train[np.random.randint(dataSet.shape[0], size=2), :] #random centres
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
        return centroids

k = 2
centroids = k_means(x_train, k)

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
            r = x[i]-centre[j]
            phi[i][j] = guassian(np.linalg.norm(r))
    return phi

def weights(x,d,centre):   
    phi = phi_matrix(x,centre)
    return np.linalg.inv(np.dot(phi.T,phi)).dot(phi.T).dot(d)

def predict(centre,x,W):
    y_pre = np.zeros((x.shape[0], 1))
    for j in range(x.shape[0]):
        y = 0
        for i in range(centre.shape[0]):
            r = x[j] - centre[i]
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

# accuracy
plt.figure()
plt.plot(Thr,TrAcc,Thr,TeAcc)
plt.title('Accuracy in K-Mean Clustering with 2 centres')
plt.legend(('Train ', 'Test'))
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
plt.clf()


mean_value = np.zeros((2,784))
for i in range(k):
    mean_value[i,:] = np.mean(x_train[np.nonzero(y_train[:,0]==i)[0]],axis =0)
    

# dimension reduction
pca = PCA(2)
projected = pca.fit_transform(centroids)
classmean = pca.fit_transform(mean_value)

plt.scatter(projected[0:,0],projected[0:,1],s=400,c='red',marker='o',alpha=0.5,label='Centre A')
plt.scatter(classmean[1:,0],classmean[1:,1],s=400,c='black',marker='o',alpha=0.5,label='Class 0 mean')

plt.scatter(projected[1:,0],projected[1:,1],s=400,c='blue',marker='o',alpha=0.5,label='Centre B')

plt.scatter(classmean[0:,0],classmean[0:,1],s=400,c='green',marker='o',alpha=0.5,label='Class 1 mean')


plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend(prop = {'size':16})
plt.show()
plt.clf()


plt.subplot(2,2,1)
plt.imshow(centroids[0].reshape(28,28).T,cmap='gray')
plt.title('Centre A')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(centroids[1].reshape(28,28).T,cmap='gray')
plt.title('Centre B')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(mean_value[1].reshape(28,28).T,cmap='gray')
plt.title('Class 1 mean')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(mean_value[0].reshape(28,28).T,cmap='gray')
plt.title('Class 0 mean')
plt.axis('off')
plt.show()
