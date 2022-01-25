#### Machine Learning with Scikit-learn ####
'''
Structure:
- imports
- data
- standardisation
- outlier detection
- validation curve, parameter estimation
    classifier.fit(data).predict(
- pipeline
- model testing



'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

iris = datasets.load_iris(as_frame=True)
print(iris.DESCR)
print(dir(iris))


## Scaling (linear) ##
# (If your data has lots of outliers a more robust scaler should be used)
# First fit the scaler to the data
scaler = StandardScaler().fit(iris.data)
#scaler.mean # shows the mean of the data
#scaler.scale # shows the scale factor of the normalised data

# Then scale the data itself
iris_scaled = scaler.transform(iris.data)
iris_scaled.mean
iris_scaled.std

pipe = make_pipeline(StandardScaler(), LogisticRegression())

train, test = train_test_split(iris, 10, random_state=0, shuffle=True) # 10
pipe.fit(train, test)
pipe.predict(iris)