# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


yorumlar = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
eklenenDegerler = yorumlar.iloc[:, -1:].values
imputer = imputer.fit(eklenenDegerler[:, -1:])


eklenenDegerler[:, -1:] = imputer.transform(eklenenDegerler[:, -1:])

sonuc1 = pd.DataFrame(data=eklenenDegerler, index= range(716),columns = ['Liked'])
review = yorumlar.iloc[:,0:1].values

sonuc2 = pd.DataFrame(data = review, index= range(716),columns = ['Review'])
yorumlar1 = pd.concat([sonuc2, sonuc1],axis=1)

import nltk
import re

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

durma = nltk.download('stopwords')

from nltk.corpus import stopwords

derlem = []
for i in range(716):
    yorum = re.sub('[^a-zA-z]',' ',yorumlar1['Review'][i]) #noktalama temizleme
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    derlem.append(yorum)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() #bagımsız degisken
y = yorumlar1.iloc[:,1].values #bagımlı degisken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


