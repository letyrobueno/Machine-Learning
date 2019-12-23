""" Load the Iris dataset with measurements for each species to apply Classification.

Scikit-learn supervised learning page: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Attribute Information of the dataset:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class:
        5.1 Iris-Setosa
        5.2 Iris-Versicolour
        5.3 Iris-Virginica

Source: https://scikit-learn.org/stable/datasets/index.html#iris-dataset
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# IRIS DATASET IN SCIKIT-LEARN
iris = datasets.load_iris()

# EXPLORATORY DATA ANALYSIS (EDA)
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())

# Scatter matrices: to easily visualize trends in the data
# pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D') # Histograms on the diagonals to display density
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10)) # Histograms on the diagonals to display density
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')# KDE: to estimate probability density function of variable
# plt.show()

# CLASSIFICATION involves: 
# 1. Fitting a classifier
# 2. Predicting on unlabeled data

# Accuracy: number of correct predictions divided by the total number of data points.
# To measure model performance we:
# (1) Split data into training and test set;
# (2) Fit/train the classifier on the training set;
# Note that: (i) parameters must be numpy arrays or pandas dataframes, and; (ii) no missing values are allowed on the data
# (3) Make predictions on test set;
# (4) Compare predictions with the known labels.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# (1) Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Larger k means smoother decision boundary which means less complex model
# Smaller k means more complex model which means possibility of overfitting
knn = KNeighborsClassifier(n_neighbors=8)

# (2) Fit/train the classifier on the training set
knn.fit(X_train, y_train)

# (3) Make predictions on test set
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# (4) Compare predictions with the known labels
print(knn.score(X_test, y_test))