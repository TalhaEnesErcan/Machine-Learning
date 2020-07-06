# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:37:53 2020

@author: Talha
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')

# veri on isleme
o = veriler.iloc[:,0:1].values
p = veriler.iloc[:,-1:].values
w = veriler.iloc[:,3:4].values

#encoder:  Kategorik -> Numeric

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
w=ohe.fit_transform(w).toarray()
print(w)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
p[:,0] = le.fit_transform(p[:,0])
print(p)


o = ohe.fit_transform(o).toarray()
print(o)

bolme1 = veriler.iloc[:,1:3].values

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = o, index = range(14), columns=['overcast','rainy','sunny'] )
print(sonuc)

bolme_w = pd.DataFrame(data = w[:,:1], index = range(14), columns=['windy'] )
print(bolme_w)


bolme2 = pd.DataFrame(data = bolme1 , index=range(14), columns=['temperature','humidity'])
print(bolme2)

bolme3 = pd.DataFrame(data = p , index=range(14), columns=['play'])
print(bolme3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,bolme2],axis=1)
print(s)

s2= pd.concat([s,bolme_w],axis=1)
print(s2)

s3= pd.concat([s2,bolme3],axis=1)
print(s3)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s2,bolme3,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#cinsiyet tahmini
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#P Value okuma ve Backward Elemination kodu

import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = s3.iloc[:,:-1], axis = 1)
X_l = s3.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = s3.iloc[:,-1:], exog = X_l)
r = r_ols.fit()
print(r.summary())







    
    

