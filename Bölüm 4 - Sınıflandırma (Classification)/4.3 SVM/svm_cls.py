
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values #bağımsız degiskenler
y = veriler.iloc[:,4:].values #bağımlı değişken


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)



    
    

