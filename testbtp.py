# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:39:44 2018

@author: saksh
"""
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import csv 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import math
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")


#from sklearn import matrics
#from sklearn.metrics import accuracy_score


filename = 'DATA_9features.csv'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")
x=data[:,0:9]
#print(x)
y=data[:,-3]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
svm = SVC(C=1000)
svm.fit(x_train,y_train)
#print('accuracy of training set{:.3} '.format(svm.score(x_train,y_train)))
#print('accuracy of training set{:.3} '.format(svm.score(x_test,y_test)))


#plt.plot(x_train.min(axis=0), 'o', label='Min')
#plt.plot(x_train.max(axis=0), 'v', label='Max')
#plt.xlabel('Feature Index')
#plt.ylabel('Feature Magnitude in Log Scale')
#plt.yscale('log')
#plt.legend(loc='upper right')

min_train = x_train.min(axis=0)
range_train = (x_train - min_train).max(axis=0)

x_train_scaled = (x_train - min_train)/range_train

#print('Minimum per feature\n{}'.format(x_train_scaled.min(axis=0)))
#print('Maximum per feature\n{}'.format(x_train_scaled.max(axis=0)))

x_test_scaled = (x_test - min_train)/range_train
f1_sc=[]
pre_sc=[]
recall_sc=[]
x_train_scaled_accuracy=[]
x_test_scaled_accuracy=[]
n=range(1,1000)
for C in n :
    svm = SVC(C=C)
    svm.fit(x_train_scaled, y_train)
    y_pred =svm.predict(x_test) 
    x_train_scaled_accuracy.append(svm.score(x_train_scaled,y_train))
    x_test_scaled_accuracy.append(svm.score(x_test_scaled,y_test))
    f1_sc.append((f1_score(y_test, y_pred,average='micro')))
    pre_sc.append((precision_score(y_test, y_pred,average='micro')))
    recall_sc.append((recall_score(y_test, y_pred,average='micro')))
    
    
plt.plot(n,x_train_scaled_accuracy,label='training set ')

plt.plot(n,x_test_scaled_accuracy,label='test set ')
print('f1_measure{:.3}'.format(max(f1_sc)))
print('precision{:.3}'.format(max(pre_sc)))
print('recall{:.3}'.format(max(recall_sc)))
#print('maximum accuracy train {:.3}'.format(max(x_train_scaled_accuracy)))
print('maximum accuracy train {:.3}'.format(max(x_test_scaled_accuracy)))
print('The accuracy on the training subset: {:.3f}'.format(svm.score(x_train_scaled, y_train)))
print('The accuracy on the test subset: {:.3f}'.format(svm.score(x_test_scaled, y_test)))

#plt.plot(x_test.min(axis=0),'o',label='Min')
#plt.plot(x_test.max(axis=0),'v',label='Max')