
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Linear Regression
from sklearn.linear_model import LinearRegression
le = LinearRegression()
X = x.values
Y = y.values
le.fit(X,Y)

plt.scatter(X,Y)
plt.plot(x,le.predict(x))
plt.show()

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X))) 

#tahminler