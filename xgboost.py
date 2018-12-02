# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:47:59 2018

@author: saksh
"""

import pandas as pd
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

data = pd.read_csv( 'C:\mldata\V2.csv',header = None,encoding = "ISO-8859-1")
data[9] = data[9].fillna(1)
y = data.loc[:,9].values
x = data.loc[:,0:7].values

test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

model1 = XGBClassifier(n_estimators= 5000,max_depth=3,n_jobs=-1,seed = 1)
model1.fit(X_train, y_train)
# make predictions for test data
y_pred1 = model1.predict_proba(X_test)


model2 = XGBClassifier(n_estimators= 5000,max_depth=4, n_jobs=-1,seed = 2)
model2.fit(X_train, y_train)
# make predictions for test data
y_pred2 = model2.predict_proba(X_test)

model3 = XGBClassifier(n_estimators= 5000,max_depth=5, n_jobs=-1,seed = 3)
model3.fit(X_train, y_train)
# make predictions for test data
y_pred3 = model3.predict_proba(X_test)


model4 = XGBClassifier(n_estimators= 5000,max_depth=2 ,n_jobs=-1,seed = 4)
model4.fit(X_train, y_train)
# make predictions for test data
y_pred4 = model4.predict_proba(X_test)

model5 = XGBClassifier(n_estimators= 5000,max_depth=6 ,n_jobs=-1,seed = 5)
model5.fit(X_train, y_train)
# make predictions for test data
y_pred5 = model5.predict_proba(X_test)

y_pred = np.mean(np.stack((y_pred1[:,1],y_pred2[:,1],y_pred3[:,1],y_pred4[:,1],y_pred5[:,1]),axis=1),axis=1)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 