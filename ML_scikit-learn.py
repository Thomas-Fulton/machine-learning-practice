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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

    #### Load data ####
    ## Numpy ##
# used in first attempt when building model
iris = datasets.load_iris()
#print(iris.DESCR)
#print(dir(iris))
#print(iris.target)
#print(iris.target_names)

#print(iris.data.shape)
#print(iris.data.ndim)
print(iris.data)

    ## Pandas ##
pddata = pd.read_csv("https://raw.githubusercontent.com/terryz1/Iris_Classification/master/data/data.csv")  # use raw link from github
pddata.head()

    ## Plot and explore data ##
pddata.describe()
# see distribution of samples in each category TODO if uneven distribution?
pddata.groupby('species').size()

#plt.interactive(True)

## Histograms
# see distribution for each feature
#plt.ioff()
n_bins = 10
plt.scatter(pddata['sepal_length'], pddata['sepal_width'])
#plt.hist(pddata['sepal_length'], bins = n_bins)
plt.show()
#hist00.set_title('Sepal Length')
#hist01 = plt.hist(pddata['sepal_width'], bins = n_bins)
#hist01.set_title('Sepal Width')
#hist10 = plt.hist(pddata['petal_length'], bins = n_bins)
#hist10.set_title('Petal Length')
#hist11 = plt.hist(pddata['petal_width'], bins = n_bins)
#hist11.set_title('Petal Width')
# add some spacing between subplots
#fig.tight_layout(pad=1.0)
#hist00.show()

# Boxplots
fig, axs = plt.subplots(2, 2)
fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
cn = ['setosa', 'versicolor', 'virginica']
sns.boxplot(x = 'species', y = 'sepal_length', data = pddata, order = cn, ax = axs[0,0]);
sns.boxplot(x = 'species', y = 'sepal_width', data = pddata, order = cn, ax = axs[0,1]);
sns.boxplot(x = 'species', y = 'petal_length', data = pddata, order = cn, ax = axs[1,0]);
sns.boxplot(x = 'species', y = 'petal_width', data = pddata,  order = cn, ax = axs[1,1]);
# add some spacing between subplots
fig.tight_layout(pad=1.0);
plt.show(block=True)

# Violin plots
sns.violinplot(x="species", y="petal_length", data=pddata, size=5, order = cn, palette = 'colorblind');
plt.show(block=True)

# Paired scatterplots
pairplot = sns.pairplot(pddata, hue="species", height = 2, palette = 'colorblind');
plt.show(block=True)

# Correlation between variables heatmap
corrmat = pddata.corr()
sns.heatmap(corrmat, annot = True, square = True);
plt.show(block=True)

# alternative
parallel_coordinates(pddata, "species", color = ['blue', 'red', 'green']);
plt.show(block=True)

    ## Check assumptions ##


    ## Scaling (linear) ##

# (If your data has lots of outliers a more robust scaler should be used)
# First fit the scaler to the data
scaler = StandardScaler().fit(iris.data)
scaler.mean_ # shows the mean of the data
print("\n\nScale factor: ", scaler.scale_)  # shows the scale factor of the normalised data

# Then scale the data itself
iris_scaled = scaler.transform(iris.data)
type(iris_scaled)

logreg = LogisticRegressionCV(max_iter=1000)

# Cross validation: return Use stratifiedKFold if the estimator is a classifier, and target is binary or multiclass (
# keeps distribution of categories in target class the same in each subset)
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