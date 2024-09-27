"""
This file is a copy of the CNN_Pipeline notebook and is used to train the model without using
jupyter notebook, since there seems to be a bug with the notebook sudently timing out. Please refer
to the notebook for well documented code.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import keras
from skimage.transform import resize

# Reading from HDF5
train_data = np.load('../../data/CNN Disaster/train_data.npz')
test_data = np.load('../../data/CNN Disaster/test_data.npz')

# Hyper parameters
epochs = 10
batch_size = 64
optimizer = "adam"

datagen = ImageDataGenerator(
        rotation_range=0,  # (degrees, 0 to 180)
        zoom_range = 0, # Randomly Zoom
        width_shift_range=0.1,  # Randomly Shift image % of width
        height_shift_range=0.1,  # Randomly Shift image % of height
        horizontal_flip=False,  # Randomly flip image
        vertical_flip=True)  # Randomly flip image

train_gen = datagen.flow(train_data["images"], train_data["labels"], batch_size=batch_size)
test_gen = datagen.flow(test_data["images"], test_data["labels"], batch_size=batch_size)

from keras.applications import ResNet50

base_model_1 = ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

#model = keras.Sequential()
#model.add(base_model_1)
#model.add(keras.layers.Dense(1, activation="sigmoid"))
#model.summary()
model = keras.models.load_model('model_9.keras')
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

model_hist = model.fit(train_gen, epochs=epochs, batch_size=batch_size, validation_data=(test_data["images"], test_data["labels"]))

np.savez('training_metrics_2.npz', accuracy=model_hist.history["accuracy"], val_accuracy=model_hist.history["val_accuracy"])
model.save('model_9.keras')