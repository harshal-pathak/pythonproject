# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:21:24 2021

@author: Harshal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris_data = pd.read_csv('iris.data', names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

print(iris_data.info())
print('\n')

iris_data['Species'] = iris_data['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
iris_data_column_list = list(iris_data.columns)
print(iris_data_column_list)
print('\n')

input_column_list = list(set(iris_data_column_list)-set(['Species']))
print(input_column_list)
print('\n')

output_list = list(['Species'])
print(output_list)
print('\n')

input_column_list.sort()
print(input_column_list)
print('\n')

scaler = StandardScaler()
iris_data[input_column_list] = scaler.fit_transform(iris_data[input_column_list])
print(iris_data)

input_data = iris_data[input_column_list]
covariance_matrix = input_data.cov()
print(covariance_matrix)

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.to_numpy())
print(eig_vals)
print(eig_vecs)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range (len(eig_vals))]

eig_pairs.sort(key = lambda x: x[0], reverse = True)

print('Eigen values in decending orders')
for i in eig_pairs:
    print(i[0],end='\n')
print('\n')
    
threshold =0.95
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

len(eig_vecs)
count

reduced_dimension = np.zeros((len(eig_vecs),count))
for i in range(count):
    reduced_dimension[:,i] = eig_pairs[i][1]
    
projected_data = iris_data[input_column_list].as_matrix().dot(reduced_dimension)
projected_dataframe = pd.DataFrame(projected_data, 
                                   columns=['Feature_1','Feature_2'])

projected_dataframe_with_class_info = pd.concat([projected_dataframe,
                                                 iris_data['Species']],axis=1)

grouped_data = projected_dataframe_with_class_info.groupby(by='Species')

for key in grouped_data.groups.key():
    class_data = grouped_data.get_group(key)
    x = class_data['Feature_1']
    y = class_data['Feature_2']
    plt.scatter(x,y)
    plt.xlabel('Feature_1')
    plt.ylabel('Feature_2')
plt.show()
