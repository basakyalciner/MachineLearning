# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:26:52 2020

@author: bskyl
"""
#Kütüphsneler:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Veri işleme:
veriler = pd.read_csv('veriler.csv')
print(veriler)

#Nominal değerler numerik değerlere çevirildi.
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()

ulke = veriler.iloc[:,0:1].values
print(ulke)

ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

cinsiyet = veriler.iloc[:,4:]
print(cinsiyet)

cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

yas = veriler.iloc[:,1:4].values
print(yas)
'''
!!NOTE: Bu dönüştürmede aklımıza dummy variable gelecek buyüzden 
yalnızca bir colonu alacağız ama belli bir değerse anlaşılması
için birden fazla colona ihtiyaç varsa hepsini alırız. Şimdi
bu verileri diğer colonlarla birleştireceğiz. 
'''
sonuc= pd.DataFrame(data=ulke,index=range(22),columns=["tr","fr","us"])
print("sonuc",sonuc)
sonuc2= pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
print("sonuc2",sonuc2)
sonuc3= pd.DataFrame(data=cinsiyet[:,:1], index=range(22),columns=["cinsiyet"])
print("sonuc3",sonuc3)

#Tablonun birleştirilme aşaması:
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)
print("s2",s2)

#Verilerin eğitim ve test için bölünmesi:
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#Verinin ölçeklendirilmesi:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)

#Burada linear regression algorithm yazıyoruz.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#Boy için bir regression yapalım:
boy = s2.iloc[:,3:4].values
print(boy)

#Bu kısımdan sonra eğitim ve test için boyun sağ ve sol kolonlarını alıp bir değere atıyoruz.
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
print(sag)

#A.values= sadece değerler alınır sutun isimleri olmaz values kullanmazsan dataframe olur:
#Bu kısımda tek bir dataframede birleştircez:
veri=  pd.concat([sol,sag],axis=1)
print(veri)

#Test & eğitim boy için:
x_test,x_train,y_test,y_train=train_test_split(veri,boy,test_size=0.33,random_state=0) 
regressor2=LinearRegression()
regressor2.fit(x_train,y_train)
y_pred1 = regressor2.predict(x_test)    

#Backward elimination kalıbı:
import statsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
#Burada elenen 1 değeri katsayı normalde x li bir ifade olmadığı için bunu ekledik.
print(X) 

X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit() #boyla ilgili olasılık değerlerimizi çıkaracak.
print(model.summary())

#P değeri 4 te büyük olduğu için eledik sonrada 5 i eledik.
X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit() #boyla ilgili olasılık değerlerimizi çıkaracak.
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit() #boyla ilgili olasılık değerlerimizi çıkaracak.
print(model.summary())

#Bu kısımdan sonra kalanla bir regressyon modeli kurup başarısına bakabilirsiniz!!!