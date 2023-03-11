# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 09:30:10 2021

@author: Harshal
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
habermans_data=pd.read_csv('haberman.data')
print(habermans_data.columns)
habermans_data.describe()
cancer_data=pd.read_csv('haberman.data',header=None,names=['age','year_of_operation','aux_node_detected','survival_status'])
cancer_data['survival_status']=cancer_data['survival_status'].map({1:'survived',2:'not_survived'})
cancer_data['survival_status']=cancer_data['survival_status'].astype('category')
print(cancer_data.head())
print('Shape:',cancer_data.shape)
print('Column:',cancer_data.columns)
print('Unique Value:',cancer_data['survival_status'].unique())
print('Value Count:',cancer_data['survival_status'].value_counts())
survived_patients = cancer_data.loc[cancer_data['survival_status'] == 'survived']
not_survived_patients = cancer_data.loc[cancer_data['survival_status'] == 'not_survived']
plt.figure(1,figsize=(14,4))
plt.subplot(121)
plt.plot(survived_patients['age'],np.zeros_like(survived_patients['age']),'o',label='survived')
plt.plot(not_survived_patients['age'],np.zeros_like(not_survived_patients['age']),'o',label='not_survived')
plt.legend()
plt.xlabel('Age')
plt.title('Histogram on the base of Age')
plt.subplot(122)
plt.plot(survived_patients['aux_node_detected'],np.zeros_like(survived_patients['aux_node_detected']),'o',label='survived')
plt.plot(not_survived_patients['aux_node_detected'],np.zeros_like(not_survived_patients['aux_node_detected']),'o',label='not_survived')
plt.legend()
plt.xlabel('Aux node deteced')
plt.show()
print('Graph based on age,aux node detected')
sns.FacetGrid(cancer_data,hue='survival_status',height=5).map(sns.distplot,'age').add_legend()
plt.title('Histogram based on survival status')
plt.show()
