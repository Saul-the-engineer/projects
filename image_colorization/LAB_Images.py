# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 22:00:17 2021

@author: saulg
"""

import pickle
import os
from tqdm import tqdm
import numpy as np
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt


def read_pickle(pickle_file, pickle_root):
    wellfile = pickle_root + pickle_file + '.pickle'
    with open(wellfile, 'rb') as handle:
        wells = pickle.load(handle)
    return wells

def Save_Pickle(Data, name:str, path:str):
    with open(path + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def LAB_Preprocess(root_dir):
    X =[]
    Y =[]
    path, dirs, files = next(os.walk(root_dir))
    for i, file in enumerate(tqdm(files)):
      try:
        img_name = os.path.join(root_dir, file)
        image = io.imread(img_name)
        image = resize(image,(256,256))
        img_lab = color.rgb2lab(image)
        X.append(img_lab [:,:,0]/100)
        Y.append(img_lab[:,:,1:]/128)
        if i % 1000 == 0:
            plt.imshow(X[len(X)-1],cmap='gray')
            plt.show()
      except Exception as e:
        print(e)
        continue
    return X, Y

path = './Data_Directory/Data/'
X, Y = LAB_Preprocess(path)
Data = {'X':X, 'Y':Y}
Save_Pickle(Data, 'Colorization_Data', './')
