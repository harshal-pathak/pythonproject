# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 20:14:08 2022

@author: Harshal
"""


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

## Exploratary Data Analysis ##

data = pd.read_csv('.\electric_motor.csv')

## Manipulating Data

print(data.head()) 
print(data.columns) # Present columns of the Data
print(data.dtypes)  # Data Type of the given Data

## Printing the information of the Data
print(data.info())

## Description of thegiven Data
print(data.describe())
# 1. total count is 103
# 2. Second row shows standard deviation of the given data
# 3. Third Row showing the min value of the given data
# 4. 4,5,6th row shows the 25%,50%,75% percent of the given data
# 5. 5th row shows the max value of the given data

## Shape of the given data
print(data.shape)

# Copying data into another frmae called data2
data2 = data.copy()

# Cross table indicating the value of how many Rapid charge category with no. of seats
pd.crosstab(data['RapidCharge'],data['Seats'],dropna=True)
# 1. 2Seater motor - 0-Rapid Charge,2-Not Rapid Charge
# 2. 4Seater motor -  19-Rapid Charge,2-Not Rapid Cahrge
# 3. 5Seater motor -  70-Rapid Charge,1-Not Rapid Charge
# 4. 6Seater motor -  3-Rapid Charge,0-Not Rapid Cahrge
# 5. 7Seater motor -  6-Rapid Charge,0-Not Raoid Charge

# Checking the Missing Value
data.isnull()
data.isnull().any()
data['FastChargeKmH'].mean()
# Mean value of the FastCharge KmH is 456.73 
#filling the mean value at missing value
data['FastChargeKmH'].fillna(data['FastChargeKmH'].mean(),inplace=True)
data['FastChargeKmH'].isnull().any()


# Creating Dummies
# Converting string into  dummy variable
print(data.columns)
data3=pd.get_dummies(data.columns,drop_first=True)

# Stat Data
data_new = data.select_dtypes(exclude=object)
print(data_new.shape)
data_stat = data_new.corr()
print(data_stat)
print(round(data_stat,2)) # Rounding with 2 value 

# data visualisation - Pair plot
df_select =  data.select_dtypes(include=['int64'])

for i in range (0, len(df_select.columns), 3):
    sns.pairplot(data=df_select,x_vars=df_select.columns[:-2],y_vars=['PriceEuro'])
    
# Price in Euro 50000 - till speed 200KmHshows max motors, single vehicle at 400KmH shows on pair plot
# 2 Seater motors are less  than 50000 Euro
# Two typeof motors 1- Rapid Charge and 2- Not Rapid Charge

### Regression Modelling ###

feature = list(set(data.columns)-set(['PriceEuro','Brand']))
target = list(['PriceEuro'])

print(target)
print(feature)

# Seperating out features andyarget
x= data.loc[:, feature].astype(float)
y= data.loc[:, target].astype(float)

# Spliting train test 30%  & 70% model
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)

# Finding the mean for the test data
base_pred = np.mean(test_y)
print(base_pred)

# Repeating the test value till length of the test value
base_pred = np.repeat(base_pred, len(test_y))
print(base_pred)

# Finding the root mean square error (RMSE)
base_root_mean_square_error = (mean_squared_error(test_y,base_pred))**0.5
print(round(base_root_mean_square_error,2))
# RMSE value of Price Euro is 39463.33

# Data Scaling
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
# RMSE value is 29627.22

plt.plot(test_y,'b*')
plt.plot(y_predicting_in_original_scale,'r*')

# Graph plotted 

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
# 29627.22 value matches with the old value

print('---%s second-----'%(time.time()-start_time))
#245Secondreqired to run the script
## END OF SCRIPT ##
