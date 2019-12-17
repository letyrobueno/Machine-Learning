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
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde') # KDE: to estimate probability density function of a variable
# plt.show()

# CLASSIFICATION involves: 
# (i) Fitting a classifier
# (ii) Predicting on unlabeled data

# To measure model performance: compute accuracy on data used to fit classifier (it's not evidence of ability to generalize)
# (1) split data into training and test set
# (2) fit/train the classifier on the training set
# Note that: (i) parameters must be numpy arrays or pandas dataframes, and; (ii) no missing values are allowed on the data
# (3) make predictions on test set
# (4) compare predictions with the known labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# (1) split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Larger k means smoother decision boundary which means less complex model
# Smaller k means more complex model which means possibility of overfitting
knn = KNeighborsClassifier(n_neighbors=8)

# (2) fit/train the classifier on the training set
knn.fit(X_train, y_train)

# (3) make predictions on test set
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# (4) compare predictions with the known labels
print(knn.score(X_test, y_test))