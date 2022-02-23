# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:38:14 2020

@author: bskyl
"""
#Eksik veriler,kategorik_veriler:
import pandas as pd
import numpy as np   
# bu kısımda elimizde bulunan ülkeleri sayısal bir veriye çevirmiş olduk makine fit
#ile öğrenip iloc ile atayıp transform ile dönüştürmüştür.
kategorik_veriler = pd.read_csv("eksikveriler.csv")
ulke = kategorik_veriler.iloc[:,0:1].values
print(ulke)
#Python’da Scikit-learn yapay öğrenme alanında en yaygın kullanılan kütüphanelerden biridir.
#Scikit-learn kütüphanesinin Preprocessing alt kütüphanesinin altında, veri ön hazırlaması 
#aşamasında kullandığımız iki encoderdan bahsedeceğim.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(kategorik_veriler.iloc[:,0])
print(ulke)
#Çıktımız:
# =============================================================================
# [['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['tr']
#  ['us']
#  ['us']
#  ['us']
#  ['us']
#  ['us']
#  ['us']
#  ['fr']
#  ['fr']
#  ['fr']
#  ['fr']
#  ['fr']
#  ['fr']
#  ['fr']]
# [[1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [2]
#  [2]
#  [2]
#  [2]
#  [2]
#  [2]
#  [0]
#  [0]
#  [0]
#  [0]
#  [0]
#  [0]
#  [0]]
# =============================================================================
#Nominal veya ordinalden bize numerik değerlere dönüş sağlar encodingler
ohe= preprocessing.OneHotEncoder()
ulke1 =ohe.fit_transform(ulke).toarray()
print(ulke1)
#Çıktımız:
# =============================================================================
# [[0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]
# =============================================================================
#Bu kısımda numpyları dataframe dönüştürdük.
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)
# =============================================================================
#  fr   tr   us
# 0   0.0  1.0  0.0
# 1   0.0  1.0  0.0
# 2   0.0  1.0  0.0
# 3   0.0  1.0  0.0
# 4   0.0  1.0  0.0
# 5   0.0  1.0  0.0
# 6   0.0  1.0  0.0
# 7   0.0  1.0  0.0
# 8   0.0  1.0  0.0
# 9   0.0  0.0  1.0
# 10  0.0  0.0  1.0
# 11  0.0  0.0  1.0
# 12  0.0  0.0  1.0
# 13  0.0  0.0  1.0
# 14  0.0  0.0  1.0
# 15  1.0  0.0  0.0
# 16  1.0  0.0  0.0
# 17  1.0  0.0  0.0
# 18  1.0  0.0  0.0
# 19  1.0  0.0  0.0
# 20  1.0  0.0  0.0
# 21  1.0  0.0  0.0 
#Çıktımız şu şekilde olacaktır. Satır sütun isimleri var dataframede, İndex yapısı var. 
# =============================================================================
yas = kategorik_veriler.iloc[:,1:4].values
sonuc2 = pd.DataFrame(data =yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)
#------------------------------------------------------------------------------
cinsiyet = kategorik_veriler.iloc[:,-1].values
print(cinsiyet)
#------------------------------------------------------------------------------
#Bu kısımda kategori olarak dataframe haline geldi satır ve sutun olarak veriler 
#index ve cinsiyet olarak başlık aldı.
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)
#------------------------------------------------------------------------------
#Bu kısımda veri kümelerini birleştirdik.
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)
#------------------------------------------------------------------------------
#Bu kısımda veri kümelerini birleştirdik.
s2=pd.concat([s,sonuc3], axis=1)
print(s2)
#------------------------------------------------------------------------------
#Bu kısımda  veri kümesini eğitim(train) ve test aşaması olarak böldük.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)
#------------------------------------------------------------------------------
#Öznitelik_Ölçekleme:
#kutuphane:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() #modulu kullanılır yaptık kolaylık.
X_train = sc.fit_transform(x_train) #donuşturme işlemini yaptık.
print(X_train) 
#sonuç verileri aynı türden bir değere dönüştürdük birbiriyle bağlantı kurabilmek için.
