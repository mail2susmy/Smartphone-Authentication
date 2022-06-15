# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:35:52 2022

@author: 91944
"""

import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from joblib import dump, load
data_leguser = pd.read_csv('user3.csv')
features = np.array(data_leguser.iloc[:,1:88])
#features = np.array(data.iloc[:,1:88])
print(features.shape)
#print(features)
labels = np.array(data_leguser['Label'])
#labels = np.array(data['Label'])

# Binarize the output
classes = list(set(labels))
y = label_binarize (labels, classes=classes)
n_classes = y.shape[1]


# with train-test  split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=labels)
#X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.00001, random_state=42, stratify=labels)
print(len(X_test))
clf = RandomForestClassifier(verbose=2,n_jobs=-1)
y_score = clf.fit(X_train,y_train)
dump(clf, 'rf_leguser1.joblib') 
#y_score = clf.fit(X_train,y_train).decision_function(X_test)
#y_score = clf.fit(X_train,y_train)
#legitimate model
#leg+imposter (10)testing 
data_leguser = pd.read_csv('imposter.csv')
features = np.array(data_leguser.iloc[:,1:88])
classes = list(set(labels))
y = label_binarize (labels, classes=classes)
n_classes = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=labels)

y_pred = clf.predict(X_test)
acc=metrics.accuracy_score(y_test, y_pred)
pr=metrics.precision_score(y_test, y_pred, average=None)
rec=metrics.recall_score(y_test, y_pred, average=None)
print("Accuracy:",acc)
print("Precision:",pr)
print("Recall:",rec)
f1=2*(pr*rec)/(pr+rec)
print("F1  : ",f1)