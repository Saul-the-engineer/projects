# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:57:15 2021

@author: saulg
"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import glob
import cv2
import os
from matplotlib.pyplot import imshow
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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

def Save_Pickle(Data, name:str, path:str):
    with open(path + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def Data_Split(df, y, Shuffle=False):
    if Shuffle:
         df = df.sample(frac=1) #The frac keyword argument specifies the fraction of rows to return in the random sample
    Y = df[y].to_frame()
    X = df.drop(y, axis=1)
    return Y, X

def Data_Join(df1, df2, method='outer', axis=1):
    return pd.concat([df1, df2], join='outer', axis=1)

def process_house_attributes(df):
    # initialize the column names of the continuous data
    continuous = ["bedrooms", "bathrooms", "area"]
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    data_continuous = cs.fit_transform(df[continuous])
    data_continuous = pd.DataFrame(data_continuous, columns=continuous)
    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    zipBinarizer = LabelBinarizer().fit_transform(df["zipcode"])
    zip_category = ['Zip_Code_'+ str(i) for i in range(zipBinarizer.shape[1])]
    zip_one_hot = pd.DataFrame(zipBinarizer, columns= zip_category)
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    X = Data_Join(data_continuous, zip_one_hot)
    # return the concatenated training and testing data
    return X

def normalize_price(df):
    cs = MinMaxScaler()
    data_continuous = cs.fit_transform(df)
    return pd.DataFrame(data_continuous, columns=df.columns)

def load_house_images(df, inputPath):
    # initialize our images array (i.e., the house images themselves)
    images = []
    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths,
        # ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")
        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32 32, and then
            # update the list of input images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)
        # tile the four input images in the output image such the first
        # image goes in the top-right corner, the second image in the
        # top-left corner, the third image in the bottom-right corner,
        # and the final image in the bottom-left corner
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]
        #imshow(outputImage)
        #plt.show()
        # add the tiled image to our set of images the network will be
        # trained on
        images.append(outputImage)
    # return our set of images
    return np.array(images)

housing_data_df = house_attributes('./Houses Dataset/HousesInfo.txt')
Y, X = Data_Split(housing_data_df, 'price')
Y = normalize_price(Y)
X = process_house_attributes(X)
imgs = load_house_images(X, './Houses Dataset/')

Data = {'X':X, 'Y':Y, 'Imgs':imgs}
Save_Pickle(Data, 'Data_Hybrid_Model', './')
#train, test = train_test_split(housing_data_df, test_size=0.33, random_state=42)
    
