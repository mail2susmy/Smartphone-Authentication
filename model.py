import pandas as pd
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


data = pd.read_csv('cleaned.csv')

features = np.array(data.drop('Label',axis=1))

labels = np.array(data['Label'])

# # with train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42, stratify=labels)

# various classifiers
# # svm
# clf = SVC(kernel = 'linear')
# # tree
# clf = DecisionTreeClassifier()
# # forest
# clf = RandomForestClassifier(verbose=2,n_jobs=-1)
# # LogisticRegression
# clf = LogisticRegression()
# # naivebayes
# clf = GaussianNB()

# fitting the model
# clf.fit(X_train,y_train)

print(classification_report(y_test,clf.predict(X_test)))


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