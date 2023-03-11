# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:13:37 2021

@author: Harshal
"""

#Import all the Library here
import time
start_time = time.time()
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Adding data for manipulation

data = pd.read_csv('.\ToyotaCorolla.csv')
data2 = data.copy()

#Printing data information and data 
print(data.info())

#Printing the description of the data
print(data.describe())

#Frequency of the  table count - Fuel - Petrol/Diesel/CNG
data['FuelType'].value_counts()

#DataType 
data.dtypes

#Cross table indicating howmany fuel type is automatic,manual
pd.crosstab(data['FuelType'],data['Automatic'],dropna=True)


#Check the missing values
data.isnull().any()
data.isnull().sum()

#If any value need to replace with mean
data['Age'].mean()
# data['Age'].fillna(data['Age'].mean(),inplace=True)

#Replacing value with mode
data['FuelType'].value_counts()
data['FuelType'].mode()
#data['FuelType].fillna(data['FuelType].mode()[0],inplace=True)

data['MetColor'].value_counts()
data['MetColor'] = data['MetColor'].astype('object')
data['MetColor'].dtypes
# Imputation using lambda function
data2= data2.apply(lambda x:x.fillna(x.mean()) if x.dtype=='int' else x.fillna(x.value_counts().index[0]))

#Droping a column
#data.drop('FuelType',axis=1,inplace=True)

# Create a new variable data without variable
data['Automatic']=data['Automatic'].astype('object')
data['MetColor']=data['MetColor'].astype('object')
data['Automatic'].value_counts()
data['MetColor'].value_counts()
#data.dtypes

data_new = data.select_dtypes(exclude='object')
print(data_new.shape)

data_stat = data_new.corr()
print(data_stat)
print(round(data_stat,2))

#Dummy variable encoding 
# Converting string into dummy variable

data=pd.get_dummies(data,drop_first=True)

# data visualisation - Pair plot
df_select =  data.select_dtypes(include=['int64'])

for i in range (0, len(df_select.columns), 3):
    sns.pairplot(data=df_select,x_vars=df_select.columns[i:i+3],y_vars=['Price'])
    
### Regression Modelling ###

features = list(set(data.columns)-set(['Price']))
target=list(['Price'])

print(target)
print(features)

# Seperating out features

x = data.loc[:, features].astype(float)
y = data.loc[:, target].astype(float)

# Spliting tarin test in 30% & 70% model

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=1)

# Finidng themean for test data
base_pred = np.mean(test_y)
print(base_pred)

# Repeating testvalue till len of test value
base_pred = np.repeat(base_pred, len(test_y))

# Find the RMSE - root mean squared error
base_root_mean_square_error = (mean_squared_error (test_y, base_pred)) **0.5
print(base_root_mean_square_error)

# data Scaling 
scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.fit(train_x)
scaler_y.fit(train_y)

# Perform standardisation by centering and scaling
train_x = scaler_x.transform(train_x)
test_x = scaler_x.transform(test_x)
train_y = scaler_y.transform(train_y)


#### Linear Regression ####
lgr= LinearRegression(fit_intercept=False)

x=lgr.fit(train_x,train_y)
x.coef_

#Predicting from x data
predict_y_lgr = lgr.predict(test_x)

# Rescaling data from test data
y_predicting_in_original_scale = scaler_y.inverse_transform(predict_y_lgr)

lr_rmse = (mean_squared_error(test_y,y_predicting_in_original_scale))**0.5
print(lr_rmse)

plt.plot(test_y,'b*')
plt.plot(y_predicting_in_original_scale,'r*')

def r_square(test_y,predicted_y):
    RSS = (np.sum(predicted_y - test_y)**2)
    TSS = np.sum((test_y - np.mean(test_y))**2)
    r_2= 1-(RSS/TSS)
    return r_2

r2_test_data= r_square(test_y,y_predicting_in_original_scale)
print(r2_test_data)
### Linear Regression using Stat Model###
import statsmodels.api as sm
model = sm.OLS(train_y,train_x).fit()
predictions = model.predict(test_x)
y_predicting_in_original_scale2 = scaler_y.inverse_transform(predictions)

model.summary()

lr_remse2 = (mean_squared_error(test_y,y_predicting_in_original_scale2))**0.5
print(lr_remse2)



print('---%s second-----'%(time.time()-start_time))
## END OF SCRIPT ##