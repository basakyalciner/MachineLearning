# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: Basak
"""
#%%Kutuphaneleri İmport Etme:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
#%%Verileri Yukleme
veriler = pd.read_csv('Dry_Bean.csv')
# print(veriler)
#%%Verilerin Egitim ve Test icin Bolunmesi:
x = veriler.iloc[:,0:16].values #bağımsız değişkenler
y = veriler.iloc[:,16:].values #bağımlı değişken
# print(y)
# print(x)
clas = veriler.iloc[:,16:].values
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)
#%%Verilerin Ölceklenmesi:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train =sc.fit_transform(x_train)
X_test = sc.transform(x_test)
#%%Logistic Regresyon Applying:
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
# print(y_pred)
# print(y_test)
#%%Boyutlandırma:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#print(cm)
#%%KNN Classifier Applying
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("cm:", cm)
#print(y_pred)
#print(y_test)
#%%Algoritmayı Değerlendirme Kısmı
#Bir algoritmayı değerlendirmek için, karışıklık matrisi, kesinlik, geri çağırma 
#ve f1 puanı en sık kullanılan metriklerdir. Bu metrikleri hesaplamak için ' nin 
#confusion_matrixve classification_reportyöntemleri sklearn.metricskullanılabilir. 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%Accuracy Calculator:
total_data=sum(cm[0])+sum(cm[1])+sum(cm[2])+sum(cm[3])+sum(cm[4])+sum(cm[5])+sum(cm[6])
false_data=sum(cm[0][1:7])+sum(cm[1][2:7])+sum(cm[2][3:7])+sum(cm[3][4:7])+sum(cm[4][5:7])+sum(cm[5][6:7])+sum(cm[6][:-1])
true_data=cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]
print("Total number of predictions:",total_data)
print("Erroneous number of predictions:",false_data)  
print("True number of predictions",true_data)
print("Brief of code:", "%",(true_data/total_data)*100, "TRUE!")
#%%Hata Oranını K Değeri ile Karşılaştırma:
    
"""Eğitim ve tahmin bölümünde, ilk seferde en iyi sonucu veren K'nin hangi değerinin 
önceden bilinmesinin bir yolu olmadığını söylemiştik. K değeri olarak rastgele 5 
seçtik ve bu sadece %92.44 doğrulukla sonuçlandı. En iyi K değerini bulmanıza 
yardımcı olmanın bir yolu, K değerinin grafiğini ve veri kümesi için karşılık gelen
hata oranını çizmektir. Bu kısımda, 1 ile 40 arasındaki tüm K değerleri için test 
setinin tahmin edilen değerleri için ortalama hatayı çizeceğiz."""

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')