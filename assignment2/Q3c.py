import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt   
import re
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras import models
from keras import layers
# image process
def getData(path,image_size):
    img_count = 0
    y =[]
    x =[]
    for img_path in tqdm(glob(path + '/*.jpg')):
        img_count +=1
        #read image 
        I = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        
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

train_path = "D:\\group_2\\train"
val_path = "D:\\group_2\\val"
img_size = 32

x_train,y_train = getData(train_path,img_size)
x_val,y_val = getData(val_path,img_size)

# Normalize data
x_train = x_train / x_train.max(axis=0)
x_val = x_val / x_val.max(axis=0)

# implement model
model = models.Sequential()
model.add(layers.Dense(16,input_dim=img_size**2,use_bias=True,activation = None))
model.add(Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,kernel_initializer = 'normal',activation='sigmoid'))
# train model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# evaluate model
history = model.fit(x_train,y_train,epochs=200,batch_size=50,validation_data=(x_val,y_val))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)


# display training and validating loss
plt.figure()
plt.subplot(1,2,1)
plt.plot(epochs,loss,'b',label='trainning loss')
plt.plot(epochs,val_loss,'r',label='validating loss')
plt.title('Training and validation loss for image size '+str(img_size)+'x'+str(img_size))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
 
# display training and validating accuracy
plt.subplot(1,2,2)
# plt.clf()
plt.plot(epochs,acc,'b',label='Training accuracy')
plt.plot(epochs,val_acc,'r',label='validating accuracy')
plt.title('Training and validating accuracy for image size '+str(img_size)+'x'+str(img_size))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
 


