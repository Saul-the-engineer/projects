# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:11:05 2021

@author: saulg
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray

#with Image.open('Elise_and_Rino.jpg') as im:
    #im.show()
img = imread('./Elise_and_Rino.jpg')
img.shape

img2 = rgb2gray(img) * 255

output = img2.tolist()
output = np.where((img2 >= 0) & (img2 < 25), '@', img2)
output = np.where((img2 >= 25) & (img2 < 25), '#', img2)
output = np.where((img2 >= 50) & (img2 < 75), '$', img2)
output = np.where((img2 >= 75) & (img2 < 100), '*', img2)
output = np.where((img2 >= 100) & (img2 < 125), '-', img2)
output = np.where((img2 >= 125) & (img2 < 150), '~', img2)
output = np.where((img2 >= 150) & (img2 < 175), '.', img2)
output = np.where((img2 >= 175), '-', img2)
