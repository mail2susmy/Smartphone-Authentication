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
data = pd.read_csv('D://PhD KTU/PhD/Smartphone Authentication Python/new.csv')


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
#features=np.array(pd.concat([a,b],axis=1))#Acc+Gyro


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
y = label_binarize (labels, classes=classes)
n_classes = y.shape[1]


# with train-test  split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=labels)

# various classifiers

# svm

# clf = OneVsRestClassifier(SVC(kernel = 'linear', verbose=True, probability=True),n_jobs=-1)
# from joblib import dump, load
# dump(clf, 'D://PhD KTU/PhD/Smartphone Authentication Python/svm_full_feature.joblib') 
# y_score = clf.fit(X_train,y_train).decision_function(X_test)

#Decision Tree

# clf = OneVsRestClassifier(DecisionTreeClassifier(),n_jobs=-1)
# from joblib import dump, load
# dump(clf, 'D://PhD KTU/PhD/Smartphone Authentication Python/DT_full_feature.joblib') 
# y_score = clf.fit(X_train,y_train).predict_proba(X_test)

#RandomForest

# clf = OneVsRestClassifier(RandomForestClassifier(verbose=2,n_jobs=-1),n_jobs=-1)
# from joblib import dump, load
# dump(clf, 'D://PhD KTU/PhD/Smartphone Authentication Python/RF_full_feature.joblib') 
# y_score = clf.fit(X_train,y_train).predict_proba(X_test)

#NB

# clf = OneVsRestClassifier(GaussianNB(),n_jobs=-1)
# from joblib import dump, load
# dump(clf, 'D://PhD KTU/PhD/Smartphone Authentication Python/NB_full_feature.joblib') 
# y_score = clf.fit(X_train,y_train).predict_proba(X_test)

#K-NN
clf = OneVsRestClassifier(KNeighborsClassifier(),n_jobs=-1)
# from joblib import dump, load
# dump(clf, 'D://PhD KTU/PhD/Smartphone Authentication Python/KNN_full_feature.joblib') 
y_score = clf.fit(X_train,y_train).predict_proba(X_test)
#clf = RandomForestClassifier(verbose=2,n_jobs=-1)
# LogisticRegression
#clf = LogisticRegression()
# # naivebayes
#clf = GaussianNB()

y_pred = clf.predict(X_test)
acc=metrics.accuracy_score(y_test, y_pred)
pr=metrics.precision_score(y_test, y_pred, average=None)
rec=metrics.recall_score(y_test, y_pred, average=None)
print("Accuracy:",acc)
print("Precision:",pr)
print("Recall:",rec)
f1=2*(pr*rec)/(pr+rec)
print("F1  : ",f1)

print(y_test.shape)

# mat = confusion_matrix(y_test, y_pred)
# print(mat)

##AUC-ROC

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds= roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve eiborsand ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(roc_auc["micro"])

#Plot AUC

plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC')
plt.legend(loc="lower right")
plt.show()
