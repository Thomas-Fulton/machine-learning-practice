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
the variable which we are aiming to predict (target: t) from the data (observed: o). The data attribute also
contains the target variables as the last column.

Logistic Regression
See the results
Look at:
check assumptions
bias of logistic regression for this dataset
residuals

'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score

iris = datasets.load_iris()
#print(iris.DESCR)
#print(dir(iris))
#print(iris.target)
#print(iris.target_names)

#print(iris.data.shape)
#print(iris.data.ndim)
print(iris.data)  # all rows (150 samples), first column

## Scaling (linear) ##
# (If your data has lots of outliers a more robust scaler should be used)
# First fit the scaler to the data
scaler = StandardScaler().fit(iris.data)
scaler.mean_ # shows the mean of the data
print(scaler.scale_) # shows the scale factor of the normalised data

# Then scale the data itself
iris_scaled = scaler.transform(iris.data)
type(iris_scaled)
print(np.around(iris_scaled, decimals=2))

logreg = LogisticRegression(C=1e5, max_iter=1000)
#pipe = make_pipeline(StandardScaler(), LogisticRegression())

k_fold = KFold(n_splits=10)
res = cross_val_score(logreg, iris_scaled, iris.target, cv=k_fold.split(iris_scaled), n_jobs=-1)
print(res)

res = [logreg.fit(iris_scaled[train], iris.target[train]).score(iris_scaled[test], iris.target[test]) for train, test in k_fold.split(iris_scaled)]
print(res)

#train_t, test = train_test_split(iris.data, iris.target, 10, random_state=0, shuffle=True) # 10
#pipe.fit(train, test)
#pipe.predict(iris)