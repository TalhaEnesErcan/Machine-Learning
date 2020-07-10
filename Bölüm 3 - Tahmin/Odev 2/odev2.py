
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

#data frame dilimleme (slice)
x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

#multiple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

import statsmodels.api as sm
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print("---------------------------")
print("Multiple Linear Regression Predict:")
print("CEO Maasi")
print(lin_reg.predict([[10]]))
print("Mudur Maasi")
print(lin_reg.predict([[7]]))
print("---------------------------")



#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
# 2. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 2)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


print("Polynomial Regression Predict:")
print("CEO Maasi")
print(lin_reg3.predict(poly_reg3.fit_transform([[10]])))
print("Mudur Maasi")
print(lin_reg3.predict(poly_reg3.fit_transform([[7]])))
print("---------------------------")


#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

#SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)
print("Scaled SVR Predict Value:")
print("CEO Maasi")
print(svr_reg.predict([[10]]))
print("Mudur Maasi")
print(svr_reg.predict([[7]]))
print("---------------------------")

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor()
r_dt.fit(X,Y)

print("Decision Tree Predict:")
print("CEO Maasi")
print(r_dt.predict([[10]]))
print("Mudur Maasi")
print(r_dt.predict([[7]]))
print("---------------------------")


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state = 0)
rf_reg.fit(X,Y)
print("Random Forest Predict:")
print("CEO Maasi")
print(rf_reg.predict([[10]]))
print("Mudur Maasi")
print(rf_reg.predict([[7]]))
print("---------------------------")


#Ozet R2 Degerleri
print('------------------------------------------')

print("Linear Regression R2 Degeri:")
print(r2_score(Y,lin_reg.predict(x.iloc[:,0:1].values)))

print('------------------------------------------')
print("Polynomial Regression R2 Degeri:")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))))

print('------------------------------------------')
print("SVR R2 Degeri:")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print('------------------------------------------')
print("Decision Tree R2 Degeri:")
print(r2_score(Y,r_dt.predict(X)))

print('------------------------------------------')
print("Random Forest R2 Degeri:")
print(r2_score(Y,rf_reg.predict(X)))
print('------------------------------------------')

