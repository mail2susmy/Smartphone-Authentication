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
data = pd.read_csv('new.csv')
import sys
#features = np.array(data.iloc[:,1:9])#DT
#features = np.array(data.iloc[:,9:16])#FT
#features = np.array(data.iloc[:,1:16])#DT+FT(without motion))
#features = np.array(data.iloc[:,16:40])#ACC
#features = np.array(data.iloc[:,40:64])#MAg
#features = np.array(data.iloc[:,64:88])#Gyro
#features = np.array(data.iloc[:,16:64])#Acc+Mag
#features = np.array(data.iloc[:,40:88])#Mag+Gyro
#features=data.drop('Label',axis=1))
#a = (data.iloc[:,16:40])#ACC
#b = (data.iloc[:,64:88])#Gyro
#features=np.array(pd.concat([a,b],axis=1))

#data=features.tolist()
#f= []
#features = np.array(data.iloc[:,16:40,64:88])#Acc+Gyro
#for i in range(len(data)):
#    f.append(data[0][16:40]+data[0]][64:88])
#features=np.array(f)
#features = np.array(data.iloc[:,16:88])#With motion
 
features = np.array(data.iloc[:,1:88])
print(features.shape)
#print(features)
labels = np.array(data['Label'])

# Binarize the output
classes = list(set(labels))
y = label_binarize(labels, classes=classes)
n_classes = y.shape[1]

#n_class = labels.shape[1]
# # with train-test  split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=labels)

# various classifiers

# # svm
# clf = OneVsRestClassifier(svm.SVC(kernel = 'linear', verbose=True, probability=True), n_jobs=-1)
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier(verbose=2,n_jobs=-1)
# LogisticRegression
# clf = LogisticRegression()
# # naivebayes
#clf = GaussianNB()
from joblib import load

#DecTree
clf1=load('DT_full_feature.joblib')        
y_score1 = clf1.fit(X_train,y_train).predict_proba(X_test)
y_pred1 = clf1.predict(X_test)

#RanFor
clf2=load('RF_full_feature.joblib')           
y_score2 = clf2.fit(X_train,y_train).predict_proba(X_test)
y_pred2 = clf2.predict(X_test)

#SVM                                                                                        
clf3=load('filename.joblib')           
y_score3 = clf3.fit(X_train,y_train).decision_function(X_test)
y_pred3 = clf3.predict(X_test)


#NB
clf4=load('BN_full_feature.joblib')
y_score4 = clf4.fit(X_train,y_train).predict_proba(X_test)
y_pred4 = clf4.predict(X_test)

#KNN

clf5=load('KNN_full_feature.joblib')
y_score5 = clf5.fit(X_train,y_train).predict_proba(X_test)
y_pred5 = clf5.predict(X_test)
# acc=metrics.accuracy_score(y_test, y_pred)
# pr=metrics.precision_score(y_test, y_pred, average=None)
# rec=metrics.recall_score(y_test, y_pred, average=None)
# print("Accuracy:",acc)
# print("Precision:",pr)
# print("Recall:",rec)
# f1=2*(pr*rec)/(pr+rec)
# print("F1  : ",f1)

# print(y_test.shape)
# print(y_score.shape)


# Compute ROC curve and ROC area for each class

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#DT
fpr1 = dict()
tpr1 = dict()
roc_auc1 = dict()
fpr1["micro"], tpr1["micro"], _ = roc_curve(y_test.ravel(), y_score1.ravel())
roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])


#RF
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()

fpr2["micro"], tpr2["micro"], _ = roc_curve(y_test.ravel(), y_score2.ravel())
roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

#SVM
fpr3 = dict()
tpr3 = dict()
roc_auc3 = dict()
fpr3["micro"], tpr3["micro"], _ = roc_curve(y_test.ravel(), y_score3.ravel())
roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])


#NB
fpr4 = dict()
tpr4 = dict()
roc_auc4 = dict()
fpr4["micro"], tpr4["micro"], _ = roc_curve(y_test.ravel(), y_score4.ravel())
roc_auc4["micro"] = auc(fpr4["micro"], tpr4["micro"])

#k-NN
fpr5 = dict()
tpr5 = dict()
roc_auc5 = dict()
fpr5["micro"], tpr5["micro"], _ = roc_curve(y_test.ravel(), y_score5.ravel())
roc_auc5["micro"] = auc(fpr5["micro"], tpr5["micro"])

#Plot AUC

plt.figure()
lw = 2
plt.plot(fpr1["micro"], tpr1["micro"], color='darkorange',
         lw=lw, label='Decision Tree (area = %f)' % roc_auc1["micro"])
plt.plot(fpr2["micro"], tpr2["micro"], color='red',
         lw=lw, label='Random Forest(area = %f)' % roc_auc2["micro"])
plt.plot(fpr3["micro"], tpr3["micro"], color='green',
         lw=lw, label='SVM (area = %f)' % roc_auc3["micro"])
plt.plot(fpr4["micro"], tpr4["micro"], color='purple',
         lw=lw, label='Naive Bayes (area = %f)' % roc_auc4["micro"])
plt.plot(fpr5["micro"], tpr5["micro"], color='purple',
         lw=lw, label='k-NN (area = %f)' % roc_auc5["micro"])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Machine Learning Model ROC Comparison')
plt.legend(loc="lower right")
plt.show()


# from scipy.optimize import brentq
# from scipy.interpolate import interp1d
# # the problem was fpr and tpr are dictionaries so we have to specify
# # the keys. In this eg i have specified key : 2. Different keys 
# # corresponds to different classes
# eer = brentq(lambda x : 1. - x - interp1d(fpr["micro"], tpr["micro"])(x), 0., 1.)
# # thresh = interp1d(fpr[2], thresholds)(eer)


# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
# 	lb = LabelBinarizer()
# 	lb.fit(y_test)
# 	y_test = lb.transform(y_test)
# 	y_pred = lb.transform(y_pred)
# 	return roc_auc_score(y_test, y_pred, average=average)