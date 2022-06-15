# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:42:41 2022

@author: 91944
"""

import os
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
#import scikitplot as skplt
from sklearn.preprocessing import label_binarize

df = pd.read_csv('user6.csv',low_memory=False)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index()
#data1 = df.values
X=df.drop(columns ='Label').astype(float)
Y=df['Label']
print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=42, stratify=Y)

model = OneClassSVM(kernel = 'rbf', gamma = 0.1)

model.fit(X_train, Y_train)

df = pd.read_csv('imposter.csv',low_memory=False)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index()
#data1 = df.values
X=df.drop(columns ='Label').astype(float)
Y=df['Label']
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.9,random_state=42, stratify=Y)

Y_pred = model.predict(X)

Acc1= metrics.accuracy_score(Y, Y_pred)
print("Accuracy ", Acc1)
#from sklearn.metrics import recall_score
#Recall1= recall_score(Y, Y_pred,average=None)
#print("Recall = %.5f ", Recall1)

#df2 = pd.read_csv('imposter.csv',low_memory=False)
#df2.head(2)
#data2 = df2.values
#X_imp=data2[:, 1:87].astype(float)
#Y_imp=data2[:,0]

#Y_pred = model.predict(X_imp)
#Acc= metrics.accuracy_score(Y_imp, Y_pred)
#print("Accuracy ", Acc)
#from sklearn.metrics import recall_score
#Recall= recall_score(Y_imp, Y_pred,average=None)
#print("Recall = %.5f ", Recall)
