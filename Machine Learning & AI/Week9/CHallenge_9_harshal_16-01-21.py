# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:09:05 2022

@author: Harshal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Importing data
data=pd.read_csv('crash_test.csv')
data.drop([36],axis=0,inplace=True)
data.set_index('S. No.',inplace=True)
data.info()

data.head(15)
data.tail(15)

for i in (data.columns):
    print(i)
    print(data[i].isnull().sum())
data=data.fillna({
    'Fine - Total Acc. - 2014': data['Fine - Total Acc. - 2014'].mean(),
    'Fine - Persons Injured - 2014': data['Fine - Persons Injured - 2014'].mean(),
    'Mist/fog - Total Acc. - 2014':data['Mist/fog - Total Acc. - 2014'].mean(),
    'Mist/fog - Persons Injured - 2014': data['Mist/fog - Persons Injured - 2014'].mean(),
    'Cloudy - Total Acc. - 2014': data['Cloudy - Total Acc. - 2014'].mean(),
    'Cloudy - Persons Killed - 2014': data['Cloudy - Persons Killed - 2014'].mean(),
    'Cloudy - Persons Injured - 2014':data['Cloudy - Persons Injured - 2014'].mean(),
    'Light rain - Total Acc. - 2014' : data['Light rain - Total Acc. - 2014'].mean(),
    'Light rain - Persons Killed - 2014': data['Light rain - Persons Killed - 2014'].mean(),
    'Light rain - Persons Injured - 2014': data['Light rain - Persons Injured - 2014'].mean(),
    'Heavy rain - Total Acc. - 2014': data['Heavy rain - Total Acc. - 2014'].mean(),
    'Heavy rain - Persons Injured - 2014': data['Heavy rain - Persons Injured - 2014'].mean(),
    'Flooding of slipways/rivulers - Total Acc. - 2014': data['Flooding of slipways/rivulers - Total Acc. - 2014'].mean(),
    'Flooding of slipways/rivulers - Persons Killed - 2014':data['Flooding of slipways/rivulers - Persons Killed - 2014'].mean(),
    'Flooding of slipways/rivulers - Persons Injured - 2014':data['Flooding of slipways/rivulers - Persons Injured - 2014'].mean(),
    'Hail/sleet - Total Acc. - 2014':data['Hail/sleet - Total Acc. - 2014'].mean(),
    'Hail/sleet - Persons Killed - 2014': data['Hail/sleet - Persons Killed - 2014'].mean(),
    'Hail/sleet - Persons Injured - 2014': data['Hail/sleet - Persons Injured - 2014'].mean(),
    'snow - Total Acc. - 2014': data['snow - Total Acc. - 2014'].mean(),
    'snow - Persons Killed - 2014': data['snow - Persons Killed - 2014'].mean(),
    'snow - Persons Injured - 2014': data['snow - Persons Injured - 2014'].mean(),
    'Strong wind - Total Acc. - 2014': data['Strong wind - Total Acc. - 2014'].mean(),
    'Strong wind - Persons Injured - 2014': data['Strong wind - Persons Injured - 2014'].mean(),
    'Dust storm - Total Acc. - 2014': data['Dust storm - Total Acc. - 2014'].mean(),
    'Dust storm - Persons Killed - 2014':data['Dust storm - Persons Killed - 2014'].mean(),
    'Dust storm - Persons Injured - 2014': data['Dust storm - Persons Injured - 2014'].mean(),
    'Very hot - Total Acc. - 2014': data['Very hot - Total Acc. - 2014'].mean(),
    'Very hot - Persons Killed - 2014':data['Very hot - Persons Killed - 2014'].mean(),
    'Very hot - Persons Injured - 2014':data['Very hot - Persons Injured - 2014'].mean(),
    'Very cold - Total Acc. - 2014':data['Very cold - Total Acc. - 2014'].mean(),
    'Very cold - Persons Killed - 2014':data['Very cold - Persons Killed - 2014'].mean(),
    'Very cold - Persons Injured - 2014':data['Very cold - Persons Injured - 2014'].mean(),
    'Other extraordinary weather condition - Total Acc. - 2014':data['Other extraordinary weather condition - Total Acc. - 2014'].mean(),
    'Other extraordinary weather condition - Persons Killed - 2014':data['Other extraordinary weather condition - Persons Killed - 2014'].mean(),
    'Other extraordinary weather condition - Persons Injured - 2014':data['Other extraordinary weather condition - Persons Injured - 2014'].mean()
    })  
    
for i in (data.columns):
    print(i)
    print(data[i].isnull().sum()) 
    
data.describe()
    
