# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:58:22 2021

@author: saulg
"""

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pc_loc = r'D:\Tensorflow\Data_Directory\RocKIE\Flood Monument East.las'


las = laspy.read(pc_loc)
print(list(las.point_format.dimension_names))
result = set(zip(las['X']*las.header.scale[0]+las.header.offset[0], 
                 las['Y']*las.header.scale[1]+las.header.offset[1], 
                 las['Z']*las.header.scale[2]+las.header.offset[2], 
                 las['red'], las['green'], las['blue']))
pc_data = pd.DataFrame(list(result), columns = ['X', 'Y', 'Z', 'R', 'G', 'B'])


# plotting points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc_data['X'], pc_data['Y'], pc_data['Z'], c= (pc_data.iloc[:,3:])/255.0)
plt.show()

pc_data.to_hdf('Tabular_Point_Cloud.h5', key='df', mode='w')
