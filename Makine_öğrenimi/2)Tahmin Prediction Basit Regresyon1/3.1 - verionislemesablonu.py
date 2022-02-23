# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme
aylar = veriler[['Aylar']]
print(aylar)
satislar = veriler[['Satislar']]
print(satislar)
satislar2 = veriler.iloc[:,:1].values
print(satislar2)
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
#verilerin ölceklenmesi:
from sklearn.preprocessing import StandardScaler
# =============================================================================
# sc=StandardScaler()
# #Normalize ettik.
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)
# =============================================================================
#Linear basit regresyon inşa etcez.
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
'''
#Bu model x_trainden y_traini öğrendi şimdi tahmin edeceğiz böylece modeli inşa etcez.
lr.fit(X_train,Y_train) 
#Not: ctrl+I ile istediğin kütüphane hakkında seçip  bilgi alabilirsin.
tahmin = lr.predict(X_test) #x_testen y_test tahmini oluşturdu. Y_test verisini görmeden yaptı.
'''
#Not: scaler kullanmadan küçük x,y ile tahmin yapabiliriz 
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test) 
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")


