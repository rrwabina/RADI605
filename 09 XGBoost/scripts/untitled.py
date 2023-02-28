# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TA09QQS0T7GvF_LBlxUci_IbJhfu7uXd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv(r'/Breast_Cancer.csv')

df.isnull().sum()

imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
for col in non_numeric_cols:
    df[col] = imputer.fit_transform(df[[col]])

missing_values = df.isnull().sum()
percent_missing = (missing_values / len(df)) * 100
features_to_drop = percent_missing[percent_missing > 30].index.tolist()
df.drop(features_to_drop, axis=1)

print(df.shape)

print(df.shape)

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
categorical_features = [col for col in df.columns if df[col].dtype == "object"]
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])
print(df.dtypes)

missing_values = df.isnull().sum()
for feature, count in missing_values.items():
    if count > 0:
        print(feature, count)

class_counts = df['diag_cancer'].value_counts()
print(class_counts)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
X = df.drop("diag_cancer", axis=1)
y = df["diag_cancer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 10]
             }
DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced", max_depth = None)
ABC = AdaBoostClassifier(base_estimator = DTC)
grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
grid_search_ABC.fit(X_train, y_train)
y_pred = grid_search_ABC.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print("Best parameters: ", grid_search_ABC.best_params_)
print("Best score: ", grid_search_ABC.best_score_)
print(classification_report(y_test, y_pred))