## Inspect, prepare and analyse data from pima indian dataset

# Adapted from Jason Brownlee's machine learning mastery in python book: https://machinelearningmastery.com/)
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# (Originally from the Diabetes and Digestive and Kidney Disease, released under the CC0: Public Domain license)


# Too many rows and algorithms may take too long to train. Too few and perhaps you do not have enough data to train
# the algorithms. Too many features and some algorithms can be distracted or suffer poor performance due
# to the curse of dimensionality.
from pandas import read_csv
from pandas import set_option
from matplotlib import pyplot

filename = "pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)

# only commented information for the first 8 lines of the file
data.info = data['preg'][0:8]
print(data.info)
data = data[9:]

print(data.shape)
print(data.dtypes)
# "preg" dtype is "object instead of "float64"
data.iloc[:, 0] = data.iloc[:, 0].astype(float, errors = 'raise')
print(data.iloc[0:8, 0])
print(data.dtypes)

# Any missing data?
nas = data.isnull().values.any()
#print(data.isnull().sum())
print("\nNas: ", nas)

# Summary of data
set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)

# Class distribution (Classification):
# are there very imbalanced number of observations in a feature? eg. only 20 diabetic out of 278
class_counts = data.groupby('class').size()
print(class_counts)

# Are your features correlated? eg. multicollinearity may affect the statistical significance of a feature in multiple
# regression models: in that case feature selection is important.
# TODO what is an accepable level of correlation for different algorithms eg. logistic regression
correlations = data.corr(method='pearson')
print(correlations)


# box and whisker plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

data.

