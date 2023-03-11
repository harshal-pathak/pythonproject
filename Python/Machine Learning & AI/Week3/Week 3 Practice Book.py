# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:52:20 2021

@author: Harshal
"""

import pandas as pd
data_csv = pd.read_csv('iris.data')

data_csv = pd.read_csv('iris.data',names=['a','b','c','d','e'],index_col=None,na_values=['??','###'])

import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScalar

wine_data = pd.read_csv('wine.data',index_col=0,na_values=['??','##','-',' '])
wine_data_col = list(wine_data.columns)


skill_lync = pd.read_excel(r'C:\Users\harsh\OneDrive\Desktop\Skill_lync Fees.xlsx',index_col=0)
skill_lync = pd.read_excel(r'C:\Users\harsh\OneDrive\Desktop\Skill_lync Fees.xlsx',index_col=0,na_values=['-','?','//','#','@'])
skill_lync.isnull().sum()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
