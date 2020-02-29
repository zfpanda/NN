import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt   
import re
import random

# image process
def getData(path,image_size):
    img_count = 0
    y =[]
    x =[]
    for img_path in tqdm(glob(path + '/*.jpg')):
        img_count +=1
        #read image 
        I = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        # I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        #resize the image
        I = cv2.resize(I,(img_size,img_size))

        x.append(np.reshape(I,[img_size*img_size,1]))
        #get labels 
        if re.findall(r'_.*?_(.*?)_.*?', img_path)[0] == '1':
            y.append(1) 
        else :
            y.append(0) 
    x_dataset = np.array(x).reshape(img_count,img_size**2).astype(float)
    y_dataset = np.array(y).reshape(img_count,1).astype(float)
    return x_dataset,y_dataset

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(x,w):
    return 1 if np.dot(w,x) > 0 else 0

def accu_rate(x,y,w):
    errors = 0
    for i in range(x.shape[0]):
        if predict(x[i],w) != y[i] :
            errors +=1
    return 1- errors/x.shape[0]

# train model to get updated weights
def train(x,y,lr):
    w = np.zeros((1,img_size**2))
    _W = w.copy()
    for j in range(epochs):
        for i in range(x.shape[0]):
            update = lr*(y[i] - sigmoid(w.dot(x[i])))[0]            
            w += update*x[i]
        _W = np.append(_W,w,axis=0)
    return _W

# image path
train_path = "D:\\group_2\\train"
val_path = "D:\\group_2\\val"
img_size =32
epochs = 100
learn_rate = 0.01

x_train,y_train = getData(train_path,img_size)
x_val,y_val = getData(val_path,img_size)
W_train = train(x_train,y_train,learn_rate)
train_accu_set = []
valid_accu_set =[]

for i in range(epochs):
    acc_train = accu_rate(x_train,y_train,W_train[i+1])
    acc_val = accu_rate(x_val,y_val,W_train[i+1])
    train_accu_set.append(acc_train)
    valid_accu_set.append(acc_val)

plt.figure()
plt.title("image size at "+str(img_size)+'x'+str(img_size) )
plt.plot(range(epochs),train_accu_set,label='train accuracy rate')
plt.plot(range(epochs),valid_accu_set,label='valid accuracy rate')
print("Accuracy for validation set is "+'%.4f' % (valid_accu_set[-1]*100) + '%')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()