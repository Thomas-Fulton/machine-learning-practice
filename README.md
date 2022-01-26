# machine-learning-practice

ideas for datasets: 
- https://www.dataquest.io/blog/free-datasets-for-projects/ 
- sequence read archive

Introduction to machine learning
- Mini-course: https://machinelearningmastery.com/python-machine-learning-mini-course/
- Pipelines: https://machinelearningmastery.com/machine-learning-modeling-pipelines/
- Quick regressions: https://machinelearningmastery.com/spot-check-regression-machine-learning-algorithms-python-scikit-learn/

Choosing an algorithm tips:
- https://www.kdnuggets.com/2020/05/guide-choose-right-machine-learning-algorithm.html
- https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/
- https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Model assumptions:
- https://www.kdnuggets.com/2021/02/machine-learning-assumptions.html

## Steps in a solving a problem through machine learning
(Jason Brownlee at https://machinelearningmastery.com/)
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
Look at the raw data: the number of rows and columns `data.head(20)` `data.shape`; which columns are features and which are labels `data.names`; is there any missing data
