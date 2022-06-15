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
data = pd.read_csv('design_new.csv')

#features = np.array(data.iloc[:,1:9])#DT
#features = np.array(data.iloc[:,9:16])#FT
features = np.array(data.iloc[:,1:16])#DT+FT(without motion))
#features = np.array(data.iloc[:,16:40])#ACC

#features = np.array(data.iloc[:,40:64])#MAg

#features = np.array(data.iloc[:,64:88])#Gyro
# features = np.array(data.iloc[:,16:64])#Acc+Mag
#features = np.array(data.iloc[:,40:88])#Mag+Gyro
# features=data.drop('Label',axis=1)# Acc+Gyro
# a = (data.iloc[:,16:40])#ACC

#b = (data.iloc[:,64:88])#Gyro
#features=np.array(pd.concat([a,b],axis=1))

#Acc+ACCM
#a = (data.iloc[:,16:40])#ACC
#b = (data.iloc[:,88:96])#Gyro
#features=np.array(pd.concat([a,b],axis=1))
#Mag+MagM
#a = (data.iloc[:,40:64])#Mag
#b = (data.iloc[:,96:104])#MagM

#Gyro+GyroM
#a = (data.iloc[:,64:88])#Gyro
#b = (data.iloc[:,104:112])#GuroM
#features=np.array(pd.concat([a,b],axis=1))
features = np.array(data.iloc[:,1:88])
# features=data.drop('Label',axis=1)


# features = np.array(data.iloc[:,16:88])#With motion
       
# features = np.array(data.iloc[:,1:114])
print(features.shape)
#print(features)
labels = np.array(data['Label'])

# # with train-test  split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# various classifiers
# # svm
# clf = SVC(kernel = 'linear')
#clf = DecisionTreeClassifier()
clf = RandomForestClassifier(verbose=2,n_jobs=-1)
# LogisticRegression
#clf = LogisticRegression()
# # naivebayes
#clf = GaussianNB()
#clf=KNeighborsClassifier(n_neighbors=15)
# clf = RandomForestClassifier(verbose=2,n_jobs=-1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc=metrics.accuracy_score(y_test, y_pred)
pr=metrics.precision_score(y_test, y_pred)
rec=metrics.recall_score(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
f1=2*(pr*rec)/(pr+rec)
print("F1  : ",f1)
# # tree
# 
# # forest
# clf = RandomForestClassifier(verbose=2,n_jobs=-1)
# # LogisticRegression
# clf = LogisticRegression()
# # naivebayes
# clf = GaussianNB()
# fitting the model
# clf.fit(X_train,y_train)
#print(metrics.classification_report(y_test,clf.predict(X_test)))
# y_pred = clf.predict(X_test)
# acc=metrics.accuracy_score(y_test, y_pred)
# hyperparameter tuning if need be
# # cross-validation and hyperparameter tuning
# pip_clf = Pipeline([
#     ('clf', SVC(kernel = 'linear'))
# ])
# parameters = {
#   'clf__kernel': ['linear','poly']
# }
# gs_clf = GridSearchCV(pip_clf, parameters, cv=5, verbose=2, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train, y_train)
# print(classification_report(y_test,gs_clf.predict(X_test)))