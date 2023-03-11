# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:22:56 2021

@author: Harshal
"""

# importing important Library
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from math import sqrt

# Read thefile
data = pd.read_csv('winequality-red.csv')

data.head()

data.shape

# Selecting the input and output for regression task
features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target = ['quality']

# Checking for any null values in dataset
data.isnull().any()

data.isnull().sum()

X = data[features]
y = data[target]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=200)

# Fit train set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction Data
y_prediction = regressor.predict(X_test)
print(y_prediction[:5])
print('*'*50)
print(y_test)

y_test.describe()

# Evaluate Linear regression accuracy using root-mean-squared error

RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred = y_prediction))
print('RMSE:',RMSE)

# Finding r square
def r_square(test_y,predicted_y):
    RSS = np.sum((test_y-predicted_y)**2)
    TSS = np.sum((test_y-np.mean(test_y))**2)
    r_2 = 1-(RSS/TSS)
    return r_2

y_test[:5]
y_prediction.shape

# Passing the arguments

r2_test_data = r_square( y_test, y_prediction)
print(r2_test_data)

r2_score(y_test,y_prediction)

y_test

def adjusted_r2_square(y,yhat):
    SS_Residual = np.sum((y-yhat)**2)
    SS_total = np.sum((y-np.mean(y))**2)
    r_squared = 1-(float(SS_Residual))/SS_total
    adjusted_r2_square = 1-(1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print(r_squared,adjusted_r2_square)
    return r_squared,adjusted_r2_square
    
r_squared,adjusted_r2_square = adjusted_r2_square(y_test,y_prediction)

# Decision Treefit a new regressor model to the training set

regressor = DecisionTreeRegressor(max_depth=50)
regressor.fit(X_train,y_train)

# Perform prdiction using decison tree regressor
y_prediction = regressor.predict(X_test)
y_prediction[:5]

# Evaluate Decision tree regressor using RMSE

RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=y_prediction))
print(RMSE)

y_test
y_prediction.shape
y_prediction1 = y_prediction.reshape(528,1)
y_prediction1.shape
y_test.shape

# Passing argument function
r2_test_data = r_square(y_test, y_prediction1)
print(r2_test_data)
r2_score(y_test,y_prediction)

def adjusted_r2_square(y,yhat):
    SS_Residual = np.sum((y-yhat)**2)
    SS_total = np.sum((y-np.mean(y))**2)
    r_squared = 1-(float(SS_Residual))/SS_total
    adjusted_r2_square = 1-(1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    print(r_squared,adjusted_r2_square)
    return r_squared,adjusted_r2_square
r_squared,adjusted_r2_square = adjusted_r2_square(y_test,y_prediction1)


# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

# Copying old dataset into new

data_classifier = data.copy()
data_classifier.head()
data_classifier['quality'].dtype

# Convert classification task
#adding label if model is greater than 6.5 good otherwise not good
# good value-1 and non good value is 0

data_classifier['quality_label']=(data_classifier['quality']>6.5)*1
data_classifier['quality_label']

# Selecting input and output lables for quality task

feature1 = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier1 = ['quality_label']

x1 = data_classifier[feature1]
y1 = data_classifier[target_classifier1]

x_train, x_test, y_train, y_test = train_test_split(x1,y1, test_size=0.33, random_state=3)

# Fit For train
wine_quality_classifier = DecisionTreeClassifier(max_leaf_nodes=20, random_state=5)
wine_quality_classifier.fit(x_train,y_train)

# Predict on test data
prediction1 = wine_quality_classifier.predict(x_test)
prediction1[:5]
print('*'*50)
print(y_test['quality_label'][:5])

# Measure accuracy of the classifer
accuracy_score(y_true=y_test,y_pred=y_prediction1)
y_test.dtypes
y1=list(y_test['quality_label'])
pred1 = list(prediction1)

# Perform metric check
confusion_matrix = confusion_matrix(y1,pred1)
print('\t','Predicted Value')
print('Original Value','\n',confusion_matrix)

# Calculating precision, recall f1 score

prec, rec, F, support = precision_recall_fscore_support(y_test, prediction1, beta=1.0)
precision = prec[1]
recall = rec[1]
F_beta = F[1]

print('Precision',precision)
print('Recall',recall)
print('F_beta',F_beta)
