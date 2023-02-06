# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:11:42 2022

@author: Harshal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt

data = pd.read_csv('.\electric_motor.csv')

print(data.head()) #Data information 

print(data.info)

print(data.describe())

data.columns

electric_column_data_list = list(data.columns)
print(electric_column_data_list)
print('\n')

# Finding null columns
data.isnull().any()

print(data['FastChargeKmH'].mean())
#Filling mean value to the missing value 
data['FastChargeKmH'].fillna(data['FastChargeKmH'].mean(),inplace=True)

# Is any null value present in column
data['FastChargeKmH'].isnull().any()

data['FastChargeKmH'].tail(25)

data=data.drop(['Brand'],axis=1)

data.columns
data.columns.isnull().any()
input_column_list = list(set(electric_column_data_list)-set(['Brand','RapidCharge','Seats','PriceEuro','RangeKm']))
input_column_list

output_list = list(['RapidCharge'])
print(output_list)

scaler = StandardScaler()
data[input_column_list]=scaler.fit_transform(data[input_column_list])
print(round(data,2))

# Computing covariance matrix
input_data = data[input_column_list]
covariance_matrix = input_data.cov()
print(round(covariance_matrix,2))

# Computing eigen value and eigenvector
eig_val,eig_vec = np.linalg.eig(covariance_matrix.to_numpy())
len(eig_val)
eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i])for i in range(len(eig_val))]

# Absolute Value
eig_pairs.sort(key = lambda x: x[0],reverse=True)
print('Eigen value in descending order\n')
for i in eig_pairs:
    print(i[0])
    
# Setting Threshold value is 95%
threshold = 0.95

# Computing number of PC's required to capture specific variance
print('Explain Variance in Percentage \n')
total_variance = 0.0
count = 0
eigv_sum = np.sum(eig_val)

for i,j in enumerate (eig_pairs):
    variance_explained = (j[0]/eigv_sum).real
    print('eigenvalue {}: {}'.format(i+1, (j[0]/eigv_sum).real*100))
    total_variance = total_variance+variance_explained
    count = count+1
    if(total_variance>=threshold):
        break
print(total_variance)

len(eig_vec)
count

# Select required Pc's based on count projection matrix w=d*k
reduced_dimension = np.zeros((len(eig_vec),count))
for i   in range(count):
    reduced_dimension[:,i] = eig_pairs[i][1]
    
# Projecting the scale data into reduced space
projected_data = data[input_column_list].to_numpy().dot(reduced_dimension)
projected_dataframe = pd.DataFrame(projected_data, columns=['PC1','PC2','PC3'])
projected_dataframe_with_class_info = pd.concat([projected_dataframe, data['RapidCharge']],axis=1)

# Calculated all the PC's