
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
import keras.utils as image
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


df = pd.read_csv("Folds.csv")
df = df.sample(frac=1)
path = "BreaKHis_v1" 

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

files_benign = getListOfFiles('/project/Breast Cancer/image/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign')
for f in files_benign:
    if f.endswith('.png'):
        
        shutil.copy(f,'augmented/benign')
files_malignant = getListOfFiles('/project/Breast Cancer/image/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant')
for f in files_malignant:
    if f.endswith('.png'):
        
        shutil.copy(f,'augmented/malignant')


benign_images = getListOfFiles('/project/Breast Cancer/image/augmented/benign')
malignent_images = getListOfFiles('/project/Breast Cancer/image/augmented/malignant')

data = pd.DataFrame(index=np.arange(0, len(benign_images)+len(malignent_images)), columns=["image", "target"])
k=0

for c in [0,1]:
        if c==1:
            for m in range(len(benign_images)):
                data.iloc[k]["image"] = benign_images[m]
                data.iloc[k]["target"] = 0
                k += 1
        else:
            for m in range(len(malignent_images)):
                data.iloc[k]["image"] = malignent_images[m]
                data.iloc[k]["target"] = 1
                k += 1

from sklearn.utils import resample
ben_upsampled = resample(data[data['target']==0],n_samples=data[data['target']==1].shape[0], random_state=42)
up_sampled = pd.concat([data[data['target']==1], ben_upsampled])
up_sampled['target'].value_counts()

import keras.utils as image
from keras.utils import np_utils, to_categorical
train_image = []
y = []


for i in tqdm(range(up_sampled.shape[0])):
    img = image.load_img(up_sampled['image'].iloc[i], target_size=(32,32,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img,img_to_array
from sklearn.model_selection import train_test_split
        

X = np.array(train_image)
y = up_sampled.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.2 , shuffle=True)

Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)
y_val=np_utils.to_categorical(y_val,2)
# print(X_train.shape)
# print(X_test.shape)
#print(X_val.shape)
model = Sequential()
#convlouton layer with the number of filters, filter size, strides steps, padding or no, activation type and the input shape.
model.add(Conv2D(30, kernel_size = (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3)))
#pooling layer to reduce the volume of input image after convolution,
model.add(MaxPool2D(pool_size=(1,1)))
#flatten layer to flatten the output
model.add(Flatten())   # flatten output of conv
model.add(Dense(150, activation='relu'))  # hidden layer of 150 neuron
model.add(Dense(2, activation='sigmoid'))  # output layer
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, Y_train, batch_size=10, epochs = 8, validation_data=(X_test, Y_test))

# history_df = pd.DataFrame(history.history)
# history_df.plot()
# plt.show()

from tensorflow.keras.models import save_model
save_model(model, "model.h5")



from tensorflow.keras.models import load_model
 
# load model
model = load_model('model.h5')









