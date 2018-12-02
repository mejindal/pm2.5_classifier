

import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# In[4]:


data=pd.read_csv('C:\mldata\V2.csv',header=None,encoding="ISO-8859-1")
y = data.loc[:,9].values
x = data.loc[:,0:7].values


# In[5]:



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[6]:


print ("Linear Kernel")
clfLinear = LinearSVC(random_state=0)
clfLinear.fit(x_train,y_train)
y_expect=y_test
y_pred= clfLinear.predict(x_test)
print ("Accuracy:", accuracy_score(y_expect,y_pred))
print('precision value={:.3}'.format(precision_score(y_test,y_pred,average="binary")))
print('recall value={:.3}'.format(recall_score(y_test,y_pred,average="binary")))
print('f1 value={:.3}'.format(f1_score(y_test,y_pred,average="binary")))
print ("\n")


# In[7]:


print ("RBF kernel")
g= [0.001, 0.01, 0.1, 1, 10, 50, 100];
for i in range(0,7):
    print ("gamma:", g[i])
    clf_rbf = SVC(kernel='rbf', gamma=g[i])
    clf_rbf.fit(x_train,y_train)
    #print(clf)
    y_expect=y_test
    y_pred=clf_rbf.predict(x_test)
    print ("Accuracy:", accuracy_score(y_expect,y_pred))
    print('precision value={:.3}'.format(precision_score(y_test,y_pred,average="binary")))
    print('recall value={:.3}'.format(recall_score(y_test,y_pred,average="binary")))
    print('f1 value={:.3}'.format(f1_score(y_test,y_pred,average="binary")))


# In[ ]:


print ("Poly kernel")
e= [2, 3];

clf_rbf = SVC(kernel='poly',degree=3)
clf_rbf.fit(x_train,y_train)
y_expect=y_test
y_pred=clf_rbf.predict(x_test)
print ("Accuracy:", accuracy_score(y_expect,y_pred))
print('precision value={:.3}'.format(precision_score(y_test,y_pred,average="binary")))
print('recall value={:.3}'.format(recall_score(y_test,y_pred,average="binary")))
print('f1 value={:.3}'.format(f1_score(y_test,y_pred,average="binary")))  
'''
clf_rbf = SVC(kernel='poly',degree=3)
clf_rbf.fit(x_train,y_train)
y_expect=y_test
y_pred=clf_rbf.predict(x_test)
print ("Accuracy:", accuracy_score(y_expect,y_pred))
'''