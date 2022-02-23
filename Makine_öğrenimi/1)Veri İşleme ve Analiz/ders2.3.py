# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#1.1 Kütüphanelerin import edilmesi:
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
#1.2 Verilerin yüklemesi:
    
veriler = pd.read_csv('veriler.csv')
print(veriler)

#-----------------------------------------------------#
#1.3 Veri on işleme:

boy = veriler[["boy"]]
print(boy)
boykilo = veriler[["boy","kilo"]]
print(boykilo)