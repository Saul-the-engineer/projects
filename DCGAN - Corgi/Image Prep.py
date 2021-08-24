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

def Classification_Image_Preprocess(root_dir, label):
    X = []
    Y = []
    path, dirs, files = next(os.walk(root_dir))
    for i, file in enumerate(tqdm(files[0:10000])):
      try:
        img_name = os.path.join(root_dir, file)
        image = io.imread(img_name)
        image = resize(image,(128,128,3))
        X.append(image)
        Y.append(label)
      except Exception as e:
        print(e)
        continue
    X = np.asarray(X, dtype=np.float32)
    return X, Y

# Image Class Datasets
path1 = r'D:\Tensorflow\Data_Directory\Corgi'

# Create Lists of datasets
X, Y = Classification_Image_Preprocess(path1, 'Corgi')

Data = {'X':X, 'Y':Y}

# Save Data
Save_Pickle(Data, 'Corgi', './')
