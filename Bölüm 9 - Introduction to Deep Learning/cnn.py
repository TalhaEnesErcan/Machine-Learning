# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 01:53:23 2020

@author: Talha
"""
# CNN / Convolutional Neural Network
 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 

classifier  = Sequential()
 
# 1) Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
 
# 2) Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
 
# Tekrar Conv & Pool
classifier.add(Convolution2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
 
# 3) Flattening
classifier.add(Flatten())
 
# 4) YSA / ANN
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=1,activation="sigmoid",))
 
# CNN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
 
# CNN ve Resimler
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
 
training_set = train_datagen.flow_from_directory("veriler/training_set",target_size=(64,64),batch_size=1,class_mode="binary")
test_set = test_datagen.flow_from_directory("veriler/test_set",target_size=(64,64),batch_size=1,class_mode="binary")
 
print("Öğreniyor")
classifier.fit_generator(training_set,steps_per_epoch=500,epochs=1,validation_data=test_set,validation_steps=203)
 
 
import numpy as np
import pandas as pd
 
test_set.reset()
pred = classifier.predict_generator(test_set,verbose=1,steps=203)
 
for i in range(0,203):
    if pred[i] > 0.5:
        pred[i] = 1
    else :
         pred[i] = 0
 
print("Prediction geçti")
print(pred)
 
test_labels = []
 
for i in range(0,203):
    test_labels.extend(np.array(test_set[i][1]))
 
print("Test Labels:")
print(test_labels)
 
dosyaisimleri = test_set.filenames
 
 
sonuc = pd.DataFrame()
sonuc["dosyaisimleri"] = dosyaisimleri
sonuc["tahminler"] = pred
sonuc["test"] = test_labels
 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels,pred)
 
print("Confussion Matrix")
print(cm)
print("Sonuç")
print(sonuc)