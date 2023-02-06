# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:11:28 2022

@author: Harshal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt

# Importing Iris Dataset

iris_data = pd.read_csv('iris.csv')

# Information of the data
iris_data.info()

# Assigning 0,1,2 to the file different part
iris_data['Species']=(iris_data['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}))

# Creating Iris data column list
iris_data_column_list =list(iris_data.columns)
print(iris_data_column_list)
print('\n')

# Dropping column ID
iris_data = iris_data.drop(['Id'],axis=1)
print(iris_data)

# Creating the list of input data column list
input_column_list =list(set(iris_data_column_list)-set(['Id','Species']))
print(input_column_list)

# Creating list of variable at output
output_list = list(['Species'])
print(output_list)

# Sorting the input list
input_column_list.sort()

# Scaling data
scaler = StandardScaler()
iris_data[input_column_list] = scaler.fit_transform(iris_data[input_column_list])
print(iris_data)

# Compute covariance using input data
input_data = iris_data[input_column_list]
covariance_matrix = input_data.cov()
print(covariance_matrix)

# Computing the eigen values and eigen vector
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix.to_numpy())
len(eig_vals)
eig_pairs= [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x:x[0], reverse=True)

# Printing eig val in decending order
for i in eig_pairs:
    print(i[0])
    
# Setting threshold as 95% variance
threshold=0.95

Total_variance =0.0
count =0 
eigv_sum = np.sum(eig_vals)

for i,j in enumerate(eig_pairs):
    variance_explained = (j[0]/eigv_sum).real
    print('Eigen Value {}:{}'.format(i+1,(j[0]/eigv_sum).real*100))
    Total_variance = Total_variance + variance_explained
    count = count+1
    
    if(Total_variance>threshold):
        break
print(Total_variance)

len(eig_vecs)
count

# Select required PC's based on model
reduced_dimension = np.zeros((len(eig_vecs),count))
for i in range (count):
    reduced_dimension[:,i]=eig_pairs[i][1]
    
# Projecting scaled data onto reduced scale
projected_data= iris_data[input_column_list].to_numpy().dot(reduced_dimension)
projected_dataframe = pd.DataFrame(projected_data,columns=['PC1','PC2'])

projected_dataframe_with_class_info = pd.concat([projected_dataframe,iris_data['Species']],axis=1)
# Grouping the projected data
grouped_data = projected_dataframe_with_class_info.groupby(by='Species')

#Plotting the projected data
for key in grouped_data.groups.keys():
    class_data = grouped_data.get_group(key)
    x = class_data['PC1']
    y = class_data['PC2']
    plt.scatter(x,y)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
plt.show()

### PCA using Machine Learning ###

PCA_Sklearn = sklearnPCA(n_components=0.95)

projected_data_sklearn = PCA_Sklearn.fit_transform(iris_data[input_column_list])

projected_data_sklearn_df = pd.DataFrame(projected_data_sklearn,columns=['PC1','PC2'])

projected_data_sklearn_df_with_class_info = pd.concat([projected_data_sklearn_df,iris_data['Species']],axis=1)

grouped_data_sklearn = projected_data_sklearn_df_with_class_info.groupby(by='Species')

for key in grouped_data_sklearn.groups.keys():
    class_data_skl = grouped_data_sklearn.get_group(key)
    x_sk = class_data_skl['PC1']
    y_sk = class_data_skl['PC2']
    plt.scatter(x_sk,y_sk)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
plt.show()