#### Machine Learning with Scikit-learn ####
'''
Structure:
- imports
- data
- standardisation
- outlier detection
- validation curve, parameter estimation
- pipeline
- model testing

Make a pipe so that different cv subsets are treated the same

A Scikit-learn estimator is a python object which applies the "fit" and "predict" methods. The "target" attribute gives
the variable which we are aiming to predict (label or target: t) from the data (features or observed: o). The data
contains the features from which to make predictions.

Logistic Regression
See the results
Look at:
check assumptions
bias of logistic regression for this dataset
residuals

Naive Bayes

Random Forest

'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

iris = datasets.load_iris()
#print(iris.DESCR)
#print(dir(iris))
#print(iris.target)
#print(iris.target_names)

#print(iris.data.shape)
#print(iris.data.ndim)
print(iris.data)

# Plot
# Check assumptions


## Scaling (linear) ##
# (If your data has lots of outliers a more robust scaler should be used)
# First fit the scaler to the data
scaler = StandardScaler().fit(iris.data)
scaler.mean_ # shows the mean of the data
print("\n\nScale factor: ", scaler.scale_) # shows the scale factor of the normalised data

# Then scale the data itself
iris_scaled = scaler.transform(iris.data)
type(iris_scaled)

logreg = LogisticRegressionCV(max_iter=1000)

# Cross validation: return
# Use stratifiedKFold if the estimator is a classifier, and target is binary or multiclass (keeps distribution of categories in target class the same in each subset)
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
res = cross_val_score(logreg, iris_scaled, iris.target, cv=k_fold.split(iris_scaled, y=iris.target), n_jobs=-1)
print("Accuracy: %.3f%% (%.3f%%)" % (res.mean()*100.0, res.std()*100.0))
print("\n", res)

res = {}
res['scores'] = [logreg.fit(iris_scaled[train], iris.target[train]).score(iris_scaled[test], iris.target[test]) for train, test
       in k_fold.split(iris_scaled, y=iris.target)]
print(res)

res['predicts'] = [logreg.fit(iris_scaled[train], iris.target[train]).predict(iris_scaled[test]) for train, test
       in k_fold.split(iris_scaled, y=iris.target)]
print(res)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


## Pipeline


#for train, test in k_fold.split(iris_scaled, y=iris.target):
#    logreg.fit(iris_scaled[train], iris.target[train]).predict(iris_scaled[test])



#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#train_t, test = train_test_split(iris.data, iris.target, 10, random_state=0, shuffle=True) # 10
#pipe.fit(train, test)
#pipe.predict(iris)






# Random Forrest classifier