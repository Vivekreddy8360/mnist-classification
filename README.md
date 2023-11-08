# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/c897a2d1-6bcc-47a2-80a4-08a55b3a2b38)


## Neural Network Model

![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/b3f67e0f-6bf2-48dd-8ae9-6431b73953df)


## DESIGN STEPS
## STEP 1: Import the required packages
## STEP 2: Load the dataset
## STEP 3: Scale the dataset
## STEP 4: Use the one-hot encoder
## STEP 5: Create the model
## STEP 6: Compile the model
## STEP 7: Fit the model
## STEP 8: Make prediction with test data and with an external data
## PROGRAM
```
Developed by: M.vivek reddy
Reg no: 212221240030
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
## Load and spliting data
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
## Image at particular index
```
single_image= X_train[59999]
single_image.shape
plt.imshow(single_image,cmap='gray')
```
## Scaling the data
```
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
## Implementing one hot encoding
```
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
## creating the model
```
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=35,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(70,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
```
## Compiling model
```
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
```
## Fitting the model
```
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
```
## OUTPUT


### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/b62b0b3d-42d0-468f-abb4-6ef3839a94d5)

![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/fd9b4f26-933a-4130-8e69-1448407db799)

### Classification Report
![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/830fdc7d-6019-41b0-ad9b-23982011b98a)


### Confusion Matrix
![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/dfcc5d33-974b-4c8c-b1f0-59f22fc56848)



### New Sample Data Prediction
![image](https://github.com/Vivekreddy8360/mnist-classification/assets/94525701/4fc467df-da39-4bb6-b37b-96d49b3af658)


## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
