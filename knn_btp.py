# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:06:28 2018

@author: saksh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import csv
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


clf = ExtraTreesClassifier()

#filename = 'DATA_9features.csv'
data = pd.read_csv( 'v1.csv',header = None,encoding = "ISO-8859-1")
y=data.loc[:,9].values
x=data.loc[:,0:8].values
data[8] = data[8].fillna(1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=45)
 
training_accuracy = []
test_accuracy = []
f1_sc=[]
pre_sc=[]
recall_sc=[]
neighbors_settings = range(1,15)
#print(neighbors_settings)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(x_train, y_train)
    print(clf)
    training_accuracy.append(clf.score(x_train, y_train))
    test_accuracy.append(clf.score(x_test, y_test))
    y_pred = clf.predict(x_test)
    f1_sc.append((f1_score(y_test, y_pred,average='micro')))
    pre_sc.append((precision_score(y_test, y_pred,average='micro')))
    recall_sc.append((recall_score(y_test, y_pred,average='micro')))

#print(f1_sc)
#print(pre_sc)
#print(recall_sc)
#from sklearn.model_selection import cross_val_score
#accs=cross_val_score(knn, x, y,cv=10,n_jobs=-1,verbose=1)
#print(mean(accs))

plt.plot(neighbors_settings,training_accuracy,label='training set ')
plt.plot(neighbors_settings,test_accuracy,label='test')
print('max test{:.3}'.format(max(test_accuracy)))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')
plt.legend()
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred1=knn.predict(x_test)
print('f1 value={:.3}'.format(f1_score(y_test,y_pred1,average="binary")))
print('precision={:.3}'.format(precision_score(y_test,y_pred1,average="binary")))
print('recall={:.3}'.format(recall_score(y_test,y_pred1,average="binary")))
#print(data[0])
