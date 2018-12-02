# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:44:55 2018

@author: saksh
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('rk.csv',header=None)
y = data.loc[:,8].values
x = data.loc[:,0:6].values
#print(y)
data[8]=data[8].fillna(1)
#print(clf.feature_importances_)

#print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
forest = RandomForestClassifier(n_estimators=5000, random_state=42,max_depth=5)
forest.fit(x_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(forest.score(x_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(forest.score(x_test, y_test)))
from sklearn.model_selection import cross_val_score
accs=cross_val_score(forest, x, y,cv=10,n_jobs=-1,verbose=1)
print(accs)


