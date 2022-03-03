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
from sklearn.preprocessing import StandardScaler, quantile_transform
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
# see distribution of samples in each category TODO if uneven distribution? https://machinelearningmastery.com/what-is-imbalanced-classification/
pddata.groupby('species').size()


## Histograms
# see distribution for each feature
#plt.ioff()
#plt.interactive(True)
n_bins = 10
fig, axs = plt.subplots(2, 2)
axs[0,0].hist(pddata['sepal_length'], bins = n_bins);
axs[0,0].set_title('Sepal Length');
axs[0,1].hist(pddata['sepal_width'], bins = n_bins);
axs[0,1].set_title('Sepal Width');
axs[1,0].hist(pddata['petal_length'], bins = n_bins);
axs[1,0].set_title('Petal Length');
axs[1,1].hist(pddata['petal_width'], bins = n_bins);
axs[1,1].set_title('Petal Width');
# add some spacing between subplots
fig.tight_layout(pad=1.0);

feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
category_names = ['setosa', 'versicolor', 'virginica']

# seaborn hist with Kernel Density Estimate:
fig, axs = plt.subplots(2, 2)
sns.histplot(pddata, x='sepal_length', hue='species', kde=True, ax = axs[0,0]);
sns.histplot(pddata, x='sepal_width', hue='species', kde=True, ax = axs[0,1]);
sns.histplot(pddata, x='petal_length', hue='species', kde=True, ax = axs[1,0]);
sns.histplot(pddata, x='petal_width', hue='species', kde=True, ax = axs[1,1]);
fig.tight_layout(pad=1.0);
plt.show(block=True)

# Boxplots
fig, axs = plt.subplots(2, 2)
sns.boxplot(x = 'species', y = 'sepal_length', data = pddata, order = category_names, ax = axs[0,0]);
sns.boxplot(x = 'species', y = 'sepal_width', data = pddata, order = category_names, ax = axs[0,1]);
sns.boxplot(x = 'species', y = 'petal_length', data = pddata, order = category_names, ax = axs[1,0]);
sns.boxplot(x = 'species', y = 'petal_width', data = pddata,  order = category_names, ax = axs[1,1]);
# add some spacing between subplots
fig.tight_layout(pad=1.0);
plt.show(block=True)

# Violin plots
sns.violinplot(x="species", y="petal_length", data=pddata, size=5, order = category_names, palette = 'colorblind');
plt.show(block=True)

# Paired scatterplots
pairplot = sns.pairplot(pddata, hue="species", height = 2, palette = 'colorblind');
plt.show(block=True)

# Correlation between variables heatmap
corrmat = pddata.corr()
sns.heatmap(corrmat, annot = True, square = True);
plt.show(block=True)
# alternative plot:
parallel_coordinates(pddata, "species", color = ['blue', 'red', 'green']);
plt.show(block=True)



    #### Check assumptions ####
# Specific to each algorithm. See check_assumptions.py (TODO)
# or notes (https://docs.google.com/document/d/16SaICaxEEwnf9FX5r6qksbG9Dh5pataqujhYIe5h8YE/edit?usp=sharing)
# eg. features must be normally distributed: plot each, statistical test (Wilcox or something).


    ## Scaling (linear) and normalisation ##
# (When using cross validation, scaling should be done on training data, and then applied to test data - this scales
# ALL data!!)
# Identify outliers with domain knowledge and eg. boxplots,  (scale first to more easily view with many features, and
# remove with threshold (or use more robust transformation):
#df = df[(df['CALI'] >= 8.5) & (df['CALI'] <= 9)]

# Scale so each feature is on the same scale, eg. 0-1:
# First fit the scaler to the data. (If your data has lots of outliers a more robust scaler should be used, or the
# outliers should be removed - standardscaler and minmaxscaler are v sensitive).
scaler = StandardScaler().fit(iris.data)
scaler.mean_  # shows the mean of the data
print("\n\nScale factor: ", scaler.scale_)  # shows the scale factor of the scaled data.

# Then scale the data itself
iris_scaled = scaler.transform(iris.data)
type(iris_scaled)

# If data should be normalised to a Gaussian distribution, check distribution with plot/statistical tests, and transform with either:
# (power_transform, like taking log10 of data), or:
scaler = quantile_transform(iris.data, output_distribution="normal").fit_transform(iris.data)
#check with plot


    #### Cross Validation ####
# Use stratifiedKFold if the estimator is a classifier, and target is binary or multiclass (keeps distribution of
# categories in target class the same in each subset)
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)


all_models = list()


    #### Decision Tree ####

mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
predictions = list()
accuracies = list()
predicted_probs = list()
for train, test in k_fold.split(iris_scaled, y=iris.target):
    # Scale and normalise: Decision trees don't need to have normalised data!

    # Fit model and make prediction:
    mod_dt.fit(iris_scaled[train], iris.target[train])
    prediction = mod_dt.predict(iris_scaled[test])
    predictions.append(prediction)
    feature_importance = mod_dt.feature_importances_

    # Assess:
    accuracy = metrics.accuracy_score(prediction, iris.target[test])
    accuracies.append(accuracy)
    predicted_prob = mod_dt.predict_proba(iris_scaled[test])
    predicted_probs.append(predicted_prob)
    print('The accuracy of the Decision Tree is {:.3f}'.format(accuracy))
    # Confusion matrix
    disp = metrics.plot_confusion_matrix(mod_dt, iris_scaled[train], iris_scaled[test],
                                         display_labels=category_names,
                                         cmap=plt.cm.Blues,
                                         normalize=None)
    disp.ax_.set_title('Decision Tree Confusion matrix, without normalization');


# Add lists of results for each CV block to dict:
dt = {}
dt['models'] = list()
dt['accuracy'] = list()
dt['predictions'] = predictions





#### Logistic Regression ####

logreg = LogisticRegressionCV(max_iter=1000)


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