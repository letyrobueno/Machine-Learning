""" Load the Iris dataset with measurements for each species to apply Classification method: multinomial logistic regression.

Scikit-learn supervised learning page: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Instances: 150 (50 in each of three classes)
Attributes: 4 numeric, predictive attributes and the class

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

df_csv = pd.DataFrame(iris.data, columns=iris.feature_names)
df_csv['target'] = pd.Series(iris.target)
print(df_csv.head())

# Download the sklearn dataset to a csv file
df_csv.to_csv('iris_species.csv', sep = ',', index = False)

# EXPLORATORY DATA ANALYSIS (EDA)
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())

# Listing the names of all columns
print('df.columns: ')
print(list(df))

# Scatter matrices: to easily visualize trends in the data
# pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D') # Histograms on the diagonals to display density
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10)) # Histograms on the diagonals to display density
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')# KDE: to estimate probability density function of variable
# plt.show()

# CLASSIFICATION involves: 
# 1. Fitting a classifier
# 2. Predicting on unlabeled data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

y = label_binarize(y, classes=[0,1,2])
n_classes = y.shape[1]

logreg = OneVsRestClassifier(LogisticRegression(random_state=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_score = logreg.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot a ROC curve for each class in a same plot
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the multi-class data of Iris Species')
plt.legend(loc='lower right')
plt.show()


# Plot a ROC curve for each class in a different plot
# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     if i==0:
#         species = 'Iris Setosa'
#     elif i==1:
#         species = 'Iris Versicolour'
#     else:
#         species = 'Iris Virginica'
#     plt.title('ROC Curve for ' + species)
#     plt.legend(loc='lower right')
#     plt.show()
