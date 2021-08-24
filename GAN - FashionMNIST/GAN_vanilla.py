# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:20 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape


def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(n_epochs)):
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
    return generator, discriminator

def data_to_image(data):
    label = data['label'].values
    image = data.drop(['label'], axis=1)
    image = np.asarray(image.values, dtype=np.float32)
    X = []
    for i, row, in enumerate(image):
        row = np.reshape(row,(28,28))
        #plt.imshow(row, cmap='gray', interpolation='none')
        #plt.show()
        X.append(row)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(label, dtype=np.float32)
    return X, Y

data_path = r'D:\Tensorflow\Data_Directory\FashionMNIST\fashion-mnist_train.csv'
data = pd.read_csv(data_path)
X, Y = data_to_image(data)
print('Done')

codings_size = 30

generator = Sequential([
    Dense(100, activation="selu", input_shape=[codings_size]),
    Dense(150, activation="selu"),
    Dense(28 * 28, activation="sigmoid"),
    Reshape([28, 28])
])
discriminator = Sequential([
    Flatten(input_shape=[28, 28]),
    Dense(150, activation="selu"),
    Dense(100, activation="selu"),
    Dense(1, activation="sigmoid")
])
gan =Sequential([generator, discriminator])

discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
generator, discriminator = train_gan(gan, dataset, batch_size, codings_size,  n_epochs=50)


###### Prediction Plot
plt.figure(figsize=(12, 12))
plt.suptitle('Sample Predictions')
for i, index in enumerate(range(9)):
    sample_vector = np.random.random((1,30))
    out = generator(sample_vector).numpy()
    out = np.resize(out, (28,28))
    plt.subplot(3,3,i+1)
    plt.imshow(out, cmap='gray', interpolation='none')
    plt.tight_layout()
plt.show()


