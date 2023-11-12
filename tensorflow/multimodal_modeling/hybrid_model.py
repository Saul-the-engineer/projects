# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 19:25:13 2021

@author: saulg
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate

def house_attributes(input_path):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']
    df = pd.read_csv(input_path, sep= " ", header = None, names= cols)
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()
    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
    return df

def read_pickle(pickle_file, pickle_root):
    wellfile = pickle_root + pickle_file + '.pickle'
    with open(wellfile, 'rb') as handle:
        wells = pickle.load(handle)
    return wells

def Data_Split(df, y, Shuffle=False):
    if Shuffle:
         df = df.sample(frac=1) #The frac keyword argument specifies the fraction of rows to return in the random sample
    Y = df[y].to_frame()
    X = df.drop(y, axis=1)
    return Y, X

def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu", use_bias=True,
                kernel_initializer='glorot_uniform'))
    model.add(Dense(4, activation="relu", use_bias=True,
                kernel_initializer='glorot_uniform'))
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64)):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model

housing_data_df = house_attributes('./Houses Dataset/HousesInfo.txt')
Price, _ = Data_Split(housing_data_df, 'price')
cs = MinMaxScaler().fit(Price)
Data = read_pickle("Data_Hybrid_Model", './')
X = Data['X']
Y = Data['Y']
imgs = Data['Imgs']/255.0

split = train_test_split(X, Y, test_size=0.20, random_state=42)
(x_train, x_test, y_train, y_test) = split
img_train, img_test = train_test_split(imgs, test_size=0.20, random_state=42)

mlp = create_mlp(x_train.shape[1])
cnn = create_cnn(64, 64, 3)
combinedInput = concatenate([mlp.output, cnn.output])
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(x=[x_train, img_train], y= y_train, 
          validation_split=0.1, epochs=200, batch_size=8, verbose=2)
# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([x_test, img_test])

diff = preds - y_test
percentDiff = (diff / y_test) * 100
absPercentDiff = np.abs(percentDiff)
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
