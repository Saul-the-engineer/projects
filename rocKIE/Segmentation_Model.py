# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:45:46 2021

@author: saulg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle

def read_pickle(pickle_file, pickle_root):
        wellfile = pickle_root + pickle_file + '.pickle'
        with open(wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
def Save_Pickle(Data, name:str, path:str):
    with open(path + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(Data, handle, protocol=4)


Raw_Lidar = pd.read_hdf('Tabular_Point_Cloud.h5')
scaler = MinMaxScaler()
scaler.fit(Raw_Lidar)
dataset = pd.DataFrame(scaler.transform(Raw_Lidar), columns=(['X','Y','Z','R','G','B']))
dataset = dataset.sort_values(by = ['X'], axis = 0)


approximate_clusters = 1000
splits = 20
samples_split = int(len(dataset)/splits)
samples_start = 0
samples_end = samples_split
max_cluster = int(approximate_clusters/splits)
split_dict = dict()
for j in tqdm(range(0, splits)):
    if j < splits: data_split = dataset.iloc[samples_start:samples_end,:]
    else: data_split = dataset.iloc[samples_start::,:]
    var_rec = list()
    for i in tqdm(range(1, max_cluster)):
        cluster_model = KMeans(n_clusters=i, max_iter=1000)
        cluster_labels = cluster_model.fit_predict(data_split.to_numpy())
        var_rec.append(cluster_model.inertia_)
    plt.plot(var_rec, '--bo') 
    plt.show()
    split_dict['split'+str(j)] = var_rec
    samples_start = samples_end
    samples_end += samples_split
   
Save_Pickle(split_dict, 'output', './')
#np.savetxt('varience_record.txt', var_rec, delimiter='\n', fmt='%s')

