# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:20 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


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

def build_generator():
    noise_shape = (100,) #1D array of size 100 (latent vector / noise)
    #Define your generator network 
    #Here we are only using Dense layers. But network can be complicated based
    #on the application. For example, you can use VGG for super res. GAN.         
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    noise = Input(shape=noise_shape)
    img = model(noise)    #Generated image
    return Model(noise, img)

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./images/mnist_%d.png" % epoch)
    plt.close()


def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

def train(X, epochs, batch_size=128, save_interval=50):
    X_train = (X.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3) 
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 100))
        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)
        # Train the discriminator on real and fake images, separately
        #Research showed that separate training is more effective. 
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    #take average loss from real and fake images. 
    #
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
        noise = np.random.normal(0, 1, (batch_size, 100)) 
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        if epoch % save_interval == 0:
            save_imgs(epoch)


data_path = r'D:\Tensorflow\Data_Directory\FashionMNIST\fashion-mnist_train.csv'
data = pd.read_csv(data_path)
X, Y = data_to_image(data)
print('Done')

#Define input image dimensions
#Large images take too much time and resources.
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(100,))   #Our random input to the generator
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
train(X, epochs=100000, batch_size=32, save_interval=1000)
generator.save('generator_model.h5')

###### Prediction Plot
plt.figure(figsize=(12, 12))
plt.suptitle('Sample Predictions')
vector = np.random.randn(100)
vector = vector.reshape(1, 100)
X = generator.predict(vector)
plt.imshow(X[0, :, :, 0], cmap='gray_r')
plt.show()


