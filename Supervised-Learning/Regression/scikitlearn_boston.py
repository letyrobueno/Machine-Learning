""" Load the Boston house-prices dataset to apply Linear Regression to estimate the price of a house.

Scikit-learn supervised learning page: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Attribute Information of the dataset:
    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's
    
Source: UCI ML housing dataset available at https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
        https://scikit-learn.org/stable/datasets/index.html#boston-dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


##### 1st OPTION (BEGIN) #####
### Loading Boston housing data from scikit-learn datasets
boston = datasets.load_boston()
### Creating feature and target arrays
X = boston.data
y = boston.target
df = pd.DataFrame(X, columns=boston.feature_names)
print('\nboston.head: \n\n', df.head())
##### 1st OPTION (END) #####

##### 2nd OPTION (BEGIN) #####
### Loading Boston housing data from a flat file
# boston = pd.read_csv('boston.csv')
### Creating feature and target arrays
# X = boston.drop('MEDV', axis=1).values    # remove the column 'MEDV' leaving all the remaining as X (features values)
# y = boston['MEDV'].values                 # take only the column 'MEDV' as y (target values)
# print('\nboston.head: \n\n', boston.head())
##### 2nd OPTION (END) #####


########## Predicting house value from a *single* feature ##########
X_rooms = X[:,5]    # take all lines from the column 5 (with the average number of rooms)
y = y.reshape(-1, 1) # keep the 1st dimension and add another dimension of size one to y
X_rooms = X_rooms.reshape(-1, 1) # keep the 1st dimension and add another dimension of size one to X_rooms

# Plotting house value versus number of rooms
plt.scatter(X_rooms, y)
plt.ylabel('Value of house / 1000 ($)')
plt.xlabel('Number of rooms')
plt.show();

""" Fitting a regression model
    Classification can be extended to solve regression problems.
"""
import numpy as np
from sklearn import linear_model

# Perform, for the loss function, OLS (Ordinary Least Squares) which is the sum of the squares of the residuals.
# It's the same as minimizing the mean squared error of the predictions on the training set.
reg = linear_model.LinearRegression()

# As for Classification, the fit method takes vectors X, y, only that in this case y is 
# expected to have floating point values instead of integer values.
reg.fit(X_rooms, y) # arguments: number of rooms and the target variable: the house price

# Draw a line between the maximum and minimum number of rooms
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

# Plot the line
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()

########## Predicting house value from *all* features ##########
from sklearn.model_selection import train_test_split

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Instantiates the regressor
reg_all = linear_model.LinearRegression()

# Fit on the training set
reg_all.fit(X_train, y_train)

# Predict on the test set
y_pred = reg_all.predict(X_test)

# Compute the R squared (default scoring method for linear regression), which quantifies
# the amount of variance in the target predicted from the features
# It's dependent on the way the data is split
reg_all.score(X_test, y_test)

""" k-fold Cross-validation (or k-fold CV): split the dataset into k folds, then use each time 
one of them as the test set and fit the model on the remaining (k-1) folds. Predict on the 
test set, and compute the metric. Use the k values of R squared to compute mean, median, etc.
The more folds we use, the more computationally expensive it is, because there are more fittings and predictings.
"""
from sklearn.model_selection import cross_val_score
reg_cv = linear_model.LinearRegression()
results = cross_val_score(reg_cv, X, y, cv=6)
print('Results: ', results)
print('np.mean(results): ', np.mean(results))