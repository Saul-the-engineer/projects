# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 22:13:39 2021

@author: saulg
Sources:
https://keras.io/guides/writing_a_training_loop_from_scratch/
https://keras.io/getting_started/intro_to_keras_for_engineers/#using-fit-with-a-custom-training-step
https://towardsdatascience.com/building-custom-callbacks-with-keras-and-tensorflow-2-85e1b79915a3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from skimage import color
from tqdm import tqdm

import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu,True)
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import Model


def read_pickle(pickle_file, pickle_root):
    wellfile = pickle_root + pickle_file + '.pickle'
    with open(wellfile, 'rb') as handle:
        wells = pickle.load(handle)
    return wells

def Data_Prep(name, root):
    Data = read_pickle(name, root)
    #X = np.array(Data['X'][0:100])
    #Y = np.array(Data['Y'][0:100])
    X = np.array(Data['X'])
    Y = np.array(Data['Y'])
    X = X.reshape(X.shape+(1,))

    split = train_test_split(X, Y, test_size=0.20, random_state=42)
    (x_train, x_val, y_train, y_val) = split
    return x_train, x_val, y_train, y_val

def plot_grey(L):
    plt.imshow(L.squeeze(), cmap='gray')
    plt.show()

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, error, model, logs=None):
        current = error
        self.model = model
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

#@tf.function
def train_step(x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch_train, training = True)
        loss = loss_fn(y_batch_train, y_pred)
    grads = tape.gradient(loss, model.trainable_weights)  
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y_batch_train, y_pred)
    return loss

#@tf.function
def test_step(x_batch_val, y_batch_val):
    y_pred = model(x_batch_val, training = False)
    loss = loss_fn(y_batch_val, y_pred)
    val_acc_metric.update_state(y_batch_val, y_pred)
    return loss

def u_net_model():
    inputs = Input((256, 256, 1))
    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c4)
    c5 = Dropout(0.4)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='glorot_uniform', padding='same')(c9)
     
    outputs = Conv2D(2, (1, 1), activation='tanh', kernel_initializer='glorot_uniform')(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def plot_rgb(L, ab):
    test_concat = tf.concat([L, ab], 3)
    test_concat = tf.squeeze(test_concat, axis=0).numpy()
    plt.imshow(color.lab2rgb(test_concat))
    plt.show()
    
def plot_history(train_error, val_error):
    plt.plot(train_error)
    plt.plot(val_error)   
    plt.show()

x_train, x_val, y_train, y_val = Data_Prep('Colorization_Data', './')

# Make Batches
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(16)

#Build the model
model = u_net_model()
model.summary()
loss_fn = MeanSquaredError()
optimizer = Adam(learning_rate = 0.001)
train_acc_metric = RootMeanSquaredError()
val_acc_metric = RootMeanSquaredError()

epochs = 500
cb = EarlyStoppingAtMinLoss(patience=10)
cb.on_train_begin()

#Error lists
train_error = []
val_error = []

for epoch in tqdm(range(epochs)):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value)))
            print("Seen so far: %d samples" % ((step + 1) * 64))
            
    train_acc = train_acc_metric.result()
    train_error.append(train_acc_metric.result().numpy())
    print("Training error over epoch: %.4f" % (train_acc_metric.result().numpy()),)
    train_acc_metric.reset_states()
    print('Validation Debug')
    
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    val_acc = val_acc_metric.result()
    val_error.append(val_acc_metric.result().numpy())
    print("Validation error: %.4f" % (val_acc,))
    val_acc_metric.reset_states()
    
    if epoch % 5 == 0:
        i = np.random.randint(0,len(x_val))
        sample_L = x_val[i].reshape(1,256,256,1) * 100
        sample_p_ab = model(sample_L/100, training=False) * 128
        sample_ab = y_val[i].reshape(1,256,256,2) * 128
        plot_history(train_error, val_error)
        plot_grey(sample_L)
        plot_rgb(sample_L, sample_p_ab)
        plot_rgb(sample_L, sample_ab)

    cb.on_epoch_end(epoch, val_acc.numpy(), model)

model.save('./Colorization_Conv.model')
end_time = time.time()



