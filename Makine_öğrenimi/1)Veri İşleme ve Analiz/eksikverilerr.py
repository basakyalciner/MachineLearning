# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:00:38 2020

@author: bskyl
"""
#Bu derste cvs dosyamızda bazı kişilerin yaş bilgileri eksik yani bulunmamakta 
#biz burada bulunan bilgileri makine öğrenimi için hazır hale getirmeye çalışacağız.

#eksik veriler:
import pandas as pd # verileri okumak için kullanılır
import numpy as np  # nümerik işlemler yapmak için kullanılır 

eksik_veriler = pd.read_csv("eksikveriler.csv") # cvs dosyasını okuyan kod.
print(eksik_veriler)
# eksik değerler için ortalama verisini yaz a
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# sayısal olan verileri sayısal olmayan verilerden ayır
yas= eksik_veriler.iloc[:,1:4].values
print(yas)
# stratejiyi uygula
imputer = imputer.fit(yas[:,1:4]) #fit eğitmek için kullanılır burada yaş değerinş öğrencek.
#bu kısımda ise uygulama kısmını yapıyor
# alınan ortalama değerleri nan yerlerine yaz
yas[:,1:4]=imputer.transform(yas[:,1:4]) #transform ise uygulamaya yarar öğrendiğini uygulayacak.
print(yas)