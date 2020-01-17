#!/usr/bin/env python
# coding: utf-8

import numpy as np
import string
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

def model_validate(model, X, y, splits=5, shuff=True):
    model_name = type(model).__name__
    
#     Held-out validation
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3)
    model.fit(X_train, Y_train)
    held_out_result = model.predict(X_test)
    target_names = np.unique(y)
    
    print("Result of held-out validation on ", model_name)
    print(metrics.classification_report(Y_test, held_out_result, target_names=target_names))
#     print(metrics.precision_score(Y_test, held_out_result, average=None)  )
    
#     K-fold cross validation
    kf = KFold(n_splits=5, shuffle=shuff)
    results = cross_val_score(model, X, y, cv=kf)
    
    print("Result of k-fold cross validation on ", model_name)
    print(results.mean())
