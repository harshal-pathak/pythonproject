# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 09:48:23 2021

@author: Harshal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# Loading the data into the variable explorer
electric_motor_data = pd.read_csv('ELectricCarData.csv')

electric_motor_data_column_list = list(electric_motor_data.columns) 
print(electric_motor_data.columns)
print('\n')

electric_motor_data['RapidCharge'] = electric_motor_data['RapidCharge'].map({'Yes':1,'No':0})

electric_motor_data = electric_motor_data.drop(['Brand','Model','PowerTrain','PlugType','BodyStyle','Segment'],axis=1)
print(electric_motor_data)
print('\n')
input_column_list = list(set(electric_motor_data_column_list)-set(['Brand','Model','PowerTrain','PlugType','BodyStyle','Segment','RapidCharge']))
print(input_column_list)
print('\n')

output_list = list(['RapidCharge'])
print(output_list)
print('\n')
input_column_list.sort()
print(input_column_list)
print('\n')


# Constructing covariance matrix

input_data = electric_motor_data[input_column_list]
covariance_matrix = input_data.cov()
print(covariance_matrix)
print('\n')


# Finding Eigen values and Eigen vectors of the matrix

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.to_numpy())
len(eig_vals)
eig_pairs =[(np.abs(eig_vals[i]),eig_vecs[:,i])for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse=True)
print('\n')


# Arranging the Eigen pairs(Eigen Value and Eigen Vector) in descending order.
for i in eig_pairs:
    print(i[0],end='\n')

print('\n')
# Selecting the number of Eigen values such that the ration of selected and total Eigen values is 0.95.
# selecting as many Eigen vector as number of Eigen values selected.

threshold = 0.95
total_variance = 0.0
count = 0
eigv_sum = np.sum(eig_vals)
for i,j in enumerate(eig_pairs):
    variance_explained = (j[0]/eigv_sum).real
    print('eigenvalue {}: {}'.format(i+1, (j[0]/eigv_sum).real*100))
    total_variance = total_variance + variance_explained
    count = count +1
   
    if (total_variance>=threshold):
        break
print('Total_Variance',total_variance)
print('\n')
len(eig_vecs)
count
