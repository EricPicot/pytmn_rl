import numpy as np
import tf2_processing
from utils import transform_target
from tensorflow.keras import datasets, layers, models
import keyboard as kb
import os
print("here", os.listdir())
data = tf2_processing.read_image("./speed_digit_data/digit_data.npy")
target = np.load("./speed_digit_data/digit_target.npy").astype(int)
target = transform_target(target).astype('float32').reshape((-1, 10))
print("here")

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 20, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=['accuracy'])

model.fit(data, target, epochs=50, verbose=1)
model.save('./models/digit_model')
