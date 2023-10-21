# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:42:39 2021

@author: saulg
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu,True)
from tensorflow.keras import callbacks
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D

def read_pickle(pickle_file, pickle_root):
    wellfile = pickle_root + pickle_file + '.pickle'
    with open(wellfile, 'rb') as handle:
        wells = pickle.load(handle)
    return wells

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def Split_Data(X, Y, split = 0.25, state = 41):
    split = train_test_split(X, Y, test_size=split, random_state=state)
    (x_Large, x_Small, y_Large, y_Small) = split
    return x_Large, x_Small, y_Large, y_Small

def Model_Training_Metrics_plot(Data):
    #Error
    plt.plot(Data['loss'])
    plt.plot(Data['val_loss'])
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('Model Error Metric')
    plt.savefig('./Error.png')
    plt.show()
    
    #Accuracy
    plt.plot(Data['accuracy'])
    plt.plot(Data['val_accuracy'])
    plt.grid(True)
    plt.gca().set_ylim(-0.25, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('Model Accuracy')
    plt.savefig('./Accuracy.png')
    plt.show()
    return

def Data_Prep(name, root):
    # Load Preprocessed Pickle File
    Data = read_pickle(name, root)
    # Determine number of unique classes
    labels = list(set(Data['Y']))
    # Encode unique string classes
    encoder = LabelEncoder()
    encoder.fit(labels)
    Data['Y'] = encoder.transform(Data['Y'])
    
    # Transform array to numpy
    #X = np.array(Data['X'][0:100])
    #Y = np.array(Data['Y'][0:100])
    X = np.array(Data['X'])
    Y = np.array(Data['Y'])
    return X, Y, encoder

X, Y, encoder = Data_Prep('DogsVWolves', './')
x_train, x_test, y_train, y_test = Split_Data(X, Y, split=0.10)
x_train, x_val, y_train, y_val = Split_Data(x_train, y_train, split=0.25)

###### Model Initialization
loss = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate = 0.003)
Model_Input = tf.keras.Input(shape=(256,256,3))
Resnet_Model = tf.keras.applications.ResNet50(include_top = False, weights="imagenet", input_tensor = Model_Input)
for layer in Resnet_Model.layers[:143]:
    layer.trainable=False
model = Sequential()
model.add(Resnet_Model)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(1080, activation = 'selu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = optimizer, loss = loss, metrics=['accuracy'])
model.summary()
        
###### Hyper Paramter Adjustments
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=25, min_delta=0.0, restore_best_weights=True)
check_point = callbacks.ModelCheckpoint(filepath='dogVwolf_weights.h5', monitor='val_loss', mode='max', save_best_only=True)
lr_decay = callbacks.LearningRateScheduler(scheduler)
history = model.fit(x_train, y_train, epochs=500, validation_data = (x_val, y_val), verbose= 2, callbacks=[lr_decay, early_stopping, check_point])

###### Error Metrics
train_accuracy = model.evaluate(x_train, y_train)
validation_accuracy = model.evaluate(x_val, y_val)
test_accuracy = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
y_pred = model.predict_classes(x_test)

###### Prediction Plot
random_indices = [np.random.randint(0, len(x_test)) for i in range(9)]
plt.figure(figsize=(12, 12))
plt.suptitle('Sample Predictions')
for i, index in enumerate(random_indices):
    label_p = np.argmax(predictions[index])
    label_p = 'Dog' if label_p==0 else 'Wolf'
    actual = 'Dog' if y_test[index]==0 else 'Wolf'
    confidence = "{:.3f}".format(max(predictions[index]))
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[index], cmap='gray', interpolation='none')
    plt.title(f"Predicted: {label_p}, \n Class: {actual} \n Probability: {confidence}")
    plt.tight_layout()
plt.show()

Model_Training_Metrics_plot(history.history)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['True Neg','False Pos','False Neg','True Pos']
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')


df_metrics = pd.DataFrame(np.array([train_accuracy, validation_accuracy, test_accuracy]), 
                      index=(['Train', 'Validation', 'Test']), columns=(['Loss','Accuracy']))


###### Save Model and Error Metrics
model.save('./Classification.model')
df_metrics.to_hdf('./_DogWolvesMetrics.h5', key='metrics', mode='w')

