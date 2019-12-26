""" Load the breast cancer Wisconsin dataset to apply Classification method: binary logistic regression.

Scikit-learn supervised learning page: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

Instances: 569
Attributes: 30 numeric, predictive attributes and the class

Attribute Information of the dataset:
    - radius (mean of distances from center to points on the perimeter)
    - texture (standard deviation of gray-scale values)
    - perimeter
    - area
    - smoothness (local variation in radius lengths)
    - compactness (perimeter^2 / area - 1.0)
    - concavity (severity of concave portions of the contour)
    - concave points (number of concave portions of the contour)
    - symmetry
    - fractal dimension (“coastline approximation” - 1)
    - class:
        * WDBC-Malignant
        * WDBC-Benign
Class Distribution: 212 - Malignant (target==0), 357 - Benign (target==1)

Source: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
        https://scikit-learn.org/stable/datasets/index.html
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# LOADING BREAST CANCER DATASET FROM SCIKIT-LEARN:
# Convert a Scikit-learn dataset to a Pandas dataset so we can see its shape and other properties
cancer = datasets.load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)
print(df.head())

# Download the sklearn dataset to a csv file
df.to_csv('breast_cancer_wisconsin.csv', sep = ',', index = False)

# EXPLORATORY DATA ANALYSIS (EDA):
print("Cancer data set shape : {}".format(df.shape))

# Listing the names of all columns
print("df.columns: ")
# print(list(df.columns.values)) 
# OR JUST:
print(list(df))

# Counting the number of Malignant and Benign cancer cases
print("\nNumber of rows with target 0 (Malignant cases):", len(df[df.target==0])) # Malignant cases
print("\nNumber of rows with target 1 (Benign cases):", len(df[df.target==1])) # Benign cases
# OR STILL:
print("\nGrouping values of target column:\n", df.groupby('target').size())

# Checking if there are missing or null values:
print(df.isnull().sum()) # Detect missing values
print(df.isna().sum()) # Detect NA values, such as None or numpy.NaN. Empty strings are not NA values

X = df.iloc[:, 1:30].values
y = df.iloc[:, 30].values

# Scatter matrices: to easily visualize trends in the data
# pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D') # Histograms on the diagonals to display density
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10)) # Histograms on the diagonals to display density
# pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde') # KDE: to estimate probability density function of a variable
# plt.show()

# CLASSIFICATION: 
# (i) Fitting a classifier
# (ii) Predicting on unlabeled data

# To measure model performance: compute accuracy on data used to fit classifier (it's not evidence of ability to generalize)
# (1) split data into training and test set
# (2) fit/train the classifier on the training set
# Note that: (i) parameters must be numpy arrays or pandas dataframes, and; (ii) no missing values are allowed on the data
# (3) make predictions on test set
# (4) compare predictions with the known labels
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# (1) split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ******* Logistic Regression *******
# (2) fit/train the classifier on the training set
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)

# (3) make predictions on test set
y_pred = logreg.predict(X_test)
print("\nTest set predictions:\n {}".format(y_pred))

# (4) compare predictions with the known labels
# To check accuracy using the confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Function to print results of each classifier algorithm
def print_results(cm, y_pred):
    print("Confusion matrix:\n", cm)
    # Accuracy: add predicted results diagonally (number of correct predictions) divided by number of predictions
    print("\nCorrect predictions: ", cm[0, 0] + cm[1, 1])
    print("Accuracy (%):", (cm[0, 0] + cm[1, 1])/len(y_pred)) # Accuracy: 95.8%
    # OR JUST:
    # print("\nAccuracy (%):", classifier.score(X_test, y_test)) # where classifier: logreg, knn, svc, etc


# To plot the ROC curve:
from sklearn.metrics import roc_curve, roc_auc_score

""" As we vary threshold, we get a series of different false positive and true positive rates
and this set of points is the ROC curve (Receiver Operating Characteristic Curve).
"""
# Function to plot ROC curve for each classifier algorithm
def plot_roc_curve(y_test, y_pred_prob, method):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr,tpr,label=method + ", auc=" + str(auc))
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(method + " ROC Curve")
    plt.show()

print("\n******* Logistic Regression *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

print("\nClassification report:\n", classification_report(y_test, y_pred))

y_pred_prob = logreg.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Logistic Regression")


# ******* k-Nearest Neighbors (KNN) *******
# (2) fit/train the classifier on the training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train) # Accuracy: 95.1%
# (3) make predictions on test set
y_pred = knn.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* k-Nearest Neighbors (KNN) *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = knn.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "k-Nearest Neighbors (KNN)")


# ******* Support Vector Machine (SVM) *******
# (2) fit/train the classifier on the training set
from sklearn.svm import SVC
svc = SVC(kernel='linear', random_state=0, probability=True)
svc.fit(X_train, y_train) # Accuracy: 97.2%
# (3) make predictions on test set
y_pred = svc.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* Support Vector Machine (SVM) *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = svc.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Support Vector Machine (SVM)")

# ******* Kernel Support Vector Machine (K-SVM) *******
# (2) fit/train the classifier on the training set
from sklearn.svm import SVC
ksvc = SVC(kernel='rbf', random_state=0, probability=True)
ksvc.fit(X_train, y_train) # Accuracy: 96.5%
# (3) make predictions on test set
y_pred = ksvc.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* Kernel Support Vector Machine (K-SVM) *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = ksvc.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Kernel Support Vector Machine (K-SVM)")


# ******* Naïve Bayes *******
# (2) fit/train the classifier on the training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train) # Accuracy: 91.6%
# (3) make predictions on test set
y_pred = gnb.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* Naïve Bayes *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = gnb.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Naïve Bayes")


# ******* Decision Trees *******
# (2) fit/train the classifier on the training set
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, y_train) # Accuracy: 95.8%
# (3) make predictions on test set
y_pred = dt.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* Decision Trees *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = dt.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Decision Trees")


# ******* Random Forest Classification *******
# (2) fit/train the classifier on the training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train) # Accuracy: 98.6%
# (3) make predictions on test set
y_pred = rf.predict(X_test)
# (4) compare predictions with the known labels
print("\n******* Random Forest Classification *******\n")
print_results(confusion_matrix(y_test, y_pred), y_pred)

y_pred_prob = rf.predict_proba(X_test)[:,1]
plot_roc_curve(y_test, y_pred_prob, "Random Forest Classification")
