# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:49:07 2020

@author: Talha
"""


#1.kutuphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano as th
import tensorflow as tf

#2.1 Veri yükleme

veriler = pd.read_csv("Churn_Modeling.csv")

ulke = veriler.iloc[:,4:5].values
#encoder: Kategorik--->Numeric

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories = "auto")
ulke = ohe.fit_transform(ulke).toarray()


cinsiyet = veriler.iloc[:,5:6].values
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])

credit = veriler.iloc[:,3:4].values
geriyekalan = veriler.iloc[:,6:13].values

#Numpy dizileri dataFrame dönüşümü

ulke_kolonu = pd.DataFrame(data = ulke, index = range(10000), columns = ["fr","gr","sp"])

geriyekalan_kalan_kolon = pd.DataFrame(data = geriyekalan, index = range(10000), columns =["Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSlary"])

cinsiyet_kolonu = pd.DataFrame(data = cinsiyet, index = range(10000), columns = ["cinsiyet"])

credit_kolonu = pd.DataFrame(data = credit, index = range(10000), columns = ["CreditScore"])





#dataFrame birleştirme işlemi

sonuc1 = pd.concat([credit_kolonu,ulke_kolonu], axis =1)
sonuc2 = pd.concat([sonuc1,cinsiyet_kolonu], axis =1)
sonuc = pd.concat([sonuc2,geriyekalan_kalan_kolon], axis =1)


X = sonuc.iloc[:,1:13].values
Y = veriler.iloc[:,13].values


#verilerin egitim ve test için bçlünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

#verilerin ölçeklenmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Yapay Sinir Ağı

from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense      #Katman Olusturma

classifier = Sequential() #ramda bir yapay sinir ağı var

#units:Kaç gizli katmanda nöron alacağını belirlemek için kullanılır


classifier.add(Dense(6,kernel_initializer='uniform',activation='relu',input_dim=11,name="layer1"))#giriş katmanı
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu', name="layer2"))#gizli katman
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid', name="layer3"))


classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.Accuracy()])

classifier.fit(X_train,y_train,epochs=50)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

