# machine-learning-practice
Created as revision of statistical/machine learning knowledge, and and development of practical use and implementation. Contains scripts with reminders of things to consider when analysing data using statistical/machine learning, and practical code that can be adapted for future projects. [(Machine learning notes)](https://docs.google.com/document/d/16SaICaxEEwnf9FX5r6qksbG9Dh5pataqujhYIe5h8YE/edit?usp=sharing)
### Main aims:
1. To increase my broader understanding of machine learning, its limitations, good practice, and the steps in a machine learning pipeline needed to produce valid/meaningful, reproducible and accurate results.
2. To be confident in model selection from the question being asked, the available data, and suitability of the data
3. To be confident in assessing datasets to see if they meet the assumptions of different models 
4. To be confident in assessing the predictions and metrics of a model 
5. To learn how to compare different models' performance and predictions
6. To create a basic framework with scikit-learn so complete pipelines with more than one model can be easily implemented (with cross-validation and no training/testing data leakage).  

###TODO:
- Bayesian statistical modelling methods
- parameter inference
- time series
- Basic neural network with pima dataset
- check_assumptions functions
- Finish steps and considerations below

## Steps and considerations when solving a problem through statistical/machine learning
(see: Jason Brownlee at https://machinelearningmastery.com/)
1. Define Problem: Investigate and characterize the problem in order to better understand
the goals of the project. 
2. Analyze Data: Use descriptive statistics and visualization to better understand the data
you have available.
3. Prepare Data: Use data transforms in order to better expose the structure of the
prediction problem to modeling algorithms.
4. Evaluate Algorithms: Design a test harness to evaluate a number of standard algorithms
on the data and select the top few to investigate further.
5. Improve Results: Use algorithm tuning and ensemble methods to get the most out of
well-performing algorithms on your data.
6. Present Results: Finalize the model, make predictions and present results.


## 1: Define Problem
What do you want to predict? Do you want to see the effect of certain factors, or perhaps the relationship between them?
What information can be used as features, and what variable do you want to predict?

## 2: Analyse data
Look at the raw data: the number of rows and columns `data.head(20)` `data.shape`; which columns are features and which are labels `data.names`; is there any missing data? Balanced class distributions? (TODO unbalanced: https://machinelearningmastery.com/what-is-imbalanced-classification/)

## 3:


### Links
ideas for datasets:
- https://www.dataquest.io/blog/free-datasets-for-projects/ 
- papers
- sequence read archive

Introduction to machine learning
- Mini-course: https://machinelearningmastery.com/python-machine-learning-mini-course/
- Pipelines: https://machinelearningmastery.com/machine-learning-modeling-pipelines/
- Quick regressions: https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/
- nice overview: https://doi.org/10.1093/aje/kwz189 and https://DOI:10.1016/j.jacc.2018.03.521
- https://towardsdatascience.com/exploring-classifiers-with-python-scikit-learn-iris-dataset-2bcb490d2e1b

Choosing an algorithm tips:
- https://www.kdnuggets.com/2020/05/guide-choose-right-machine-learning-algorithm.html
- https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/
- https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Model assumptions:
- https://www.kdnuggets.com/2021/02/machine-learning-assumptions.html



Questions:
