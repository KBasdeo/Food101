# Classifying images from CIFAR 10 dataset

import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from numpy.random import seed

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from sklearn.metrics import confusion_matrix, classification_report

seed(888)
tf.random.set_seed(404)

# Constants
LOG_DIR = 'tensorboarad_cifar_logs/'
LABEL_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
TOTAL_INPUTS = IMAGE_PIXELS * COLOR_CHANNELS
VALIDATION_DATA_SIZE = 10000
SMALL_TRAIN_SIZE = 1000

# Getting Data
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

# Data Exploration

plt.figure(figsize=(15, 5))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train_all[i])
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(LABEL_NAMES[y_train_all[i][0]], fontsize=15)
    plt.imshow(x_train_all[i])

# plt.show()

# Image formats
nmbr_images, x, y, c = x_train_all.shape
print(f'images = {nmbr_images} \t| width = {x} \t| height = {y} \t| channels = {c}')

# Preprocessing Data
# Dividing by 255 in order to scale all data to a number between 0 and 1 (rgb scale goes up to 255)
# Also helps with learning rate. It will be easier to calculate the loss and adjusting weights
x_train_all, x_test = x_train_all / 255.0, x_test / 255.0

# Flattening data to one vector
x_train_all = x_train_all.reshape(x_train_all.shape[0], TOTAL_INPUTS)
x_test = x_test.reshape(len(x_test), TOTAL_INPUTS)

# Create validation dataset
# Instead of train/test split, validation dataset is introduced to fine tune the model before testing it on test data
# This ensures the test data only sees our best model, and makes sure our model can be applied to all data and isn't
# just fine-tuned to the test data only.
x_val = x_train_all[:VALIDATION_DATA_SIZE]
y_val = y_train_all[:VALIDATION_DATA_SIZE]

x_train = x_train_all[VALIDATION_DATA_SIZE:]
y_train = y_train_all[VALIDATION_DATA_SIZE:]

# Creating a small dataset (for illustration) prior to using the larger one
x_train_xs = x_train_all[:SMALL_TRAIN_SIZE]
y_train_xs = y_train_all[:SMALL_TRAIN_SIZE]

# Define the neural network using Keras
model_1 = Sequential([
    Dense(units=128, input_dim=TOTAL_INPUTS, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])

model_1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model_2 = Sequential()
model_2.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)))
model_2.add(Dense(128, activation='relu'))
model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(16, activation='relu'))
model_2.add(Dense(10, activation='softmax'))

model_2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model_3 = Sequential()
model_3.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)))
model_3.add(Dense(128, activation='relu'))
model_3.add(Dropout(0.25, seed=42))
model_3.add(Dense(64, activation='relu'))
model_3.add(Dense(16, activation='relu'))
model_3.add(Dense(10, activation='softmax'))

model_3.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)

# Fitting Model
samples_per_batch = 1000
nmbr_epochs = 100
# model_1.fit(x_train_xs, y_train_xs, batch_size=samples_per_batch, epochs=nmbr_epochs,
#             validation_data=(x_val, y_val))
#
# model_2.fit(x_train_xs, y_train_xs, batch_size=samples_per_batch, epochs=nmbr_epochs,
#             validation_data=(x_val, y_val))

model_3.fit(x_train, y_train, batch_size=samples_per_batch, epochs=nmbr_epochs,
            validation_data=(x_val, y_val), callbacks=[stopping], verbose=0)

# Model Evaluation
predictions = model_3.predict_classes(x_test)

print(classification_report(y_true=y_test, y_pred=predictions))
print(confusion_matrix(y_true=y_test, y_pred=predictions))

