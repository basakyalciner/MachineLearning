# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:31:24 2020

@author: bskyl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Veri işleme:
#Veri yükleme:
veriler = pd.read_csv('maaslar.csv')
print(veriler)
#--------- veriler üstünde dataframe oluşturma -------------------#
#Eğitim seviyesi:
x= veriler.iloc[:,1:2]
#Numpay array dönüşümü
X= x.values
#Maaslar:
y= veriler.iloc[:,2:3]
#Numpay array dönüşümü
Y= y.values
print(x)
print(y)
#----------------------------------------------------------------#

#Bu kısımda eğitim seviyesi ile maas arasındaki polinomial ilişkiye bakacağız:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#Train aşaması:
lin_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

#Polinomial regression:
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(X) 
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()
#-----------------------------------------------------------------#

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

#Train aşaması:
lin_reg.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

#Polinomial regression:
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #!!!!!!!!!!

x_poly = poly_reg.fit_transform(X) 
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#Tahmin:
    
print("tahmini maas:",lin_reg.predict([[11]]))
print("tahmini maas:",lin_reg.predict([[6.6]]))

print("tahmini maas:",lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print("tahmini maas:",lin_reg2.predict(poly_reg.fit_transform([[11]])))