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
    noise = Input(shape=noise_shape)
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
    model.add(Dense(np.prod(img_shape), activation='sigmoid'))
    model.add(Reshape(img_shape))
    model.summary()
    img = model(noise)    #Generated image
    return Model(inputs = noise, outputs = img, name = 'generator')

def plot_loss(des_loss, gen_loss):
    plt.plot(des_loss)
    plt.plot(gen_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('Model Error Metric')
    plt.savefig('./Error.png')
    plt.show()

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
    plt.show()


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
    return Model(inputs = img, outputs = validity, name = 'discriminator')

def train(X, batch_size=128, n_epochs= 50, save_interval=10):
    descriminator_loss_history = []
    generator_loss_history = [] 
    X_train = X.astype(np.float32)/ 255
    X_train = np.expand_dims(X_train, axis=3)
    half_batch = int(batch_size / 2)
    
    for epoch in tqdm(range(n_epochs)):
        dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1000)
        dataset = dataset.batch(half_batch, drop_remainder=True).prefetch(1)
        for X_batch in dataset:
            batch_gen_loss = []
            batch_des_loss = []
            noise = np.random.normal(0, 1, (half_batch, 100))
            # Generate a half batch of fake images
            gen_imgs = generator.predict(noise)
            # Train the discriminator on real and fake images, separately
            #Research showed that separate training is more effective. 
            descriminator_loss_real = discriminator.train_on_batch(X_batch, np.ones((half_batch, 1)))
            descriminator_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            #take average loss from real and fake images. 
            d_loss = 0.5 * np.add(descriminator_loss_real, descriminator_loss_fake)
            batch_des_loss.append(d_loss)
            
            noise = np.random.normal(0, 1, (batch_size, 100)) 
            valid_y = np.array([1] * batch_size)
            generator_loss = GAN.train_on_batch(noise, valid_y)
            batch_gen_loss.append(generator_loss)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], generator_loss))

        avg_gen_loss = sum(batch_gen_loss)/len(batch_gen_loss)
        avg_des_loss = sum(batch_des_loss[0])/len(batch_des_loss)
        descriminator_loss_history.append(avg_des_loss)
        generator_loss_history.append(avg_gen_loss)
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, avg_gen_loss, 100*d_loss[1], avg_des_loss))
        if epoch % save_interval == 0:
            plot_loss(descriminator_loss_history, generator_loss_history)
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
optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

z = Input(shape=(100,))   #Our random input to the generator
img = generator(z)

discriminator.trainable = False
valid = discriminator(img)

GAN = Model(z, valid)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
train(X, n_epochs=2, batch_size=128, save_interval=1)
generator.save('generator_model.h5')

###### Prediction Plot
plt.figure(figsize=(12, 12))
plt.suptitle('Sample Predictions')
vector = np.random.randn(100)
vector = vector.reshape(1, 100)
X = generator.predict(vector)
plt.imshow(X[0, :, :, 0], cmap='gray_r')
plt.show()


