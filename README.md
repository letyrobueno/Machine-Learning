# Machine Learning Algorithms

Conventions:
* features = predictor variables = independent variables;
* target variable = dependent variable = response variable;
* training a model on the data = fitting a model to the data.

## Supervised Learning
It uses labeled data as input, which are usually represented in a table structure. It divides in:
1. **Classification:** outputs a label, which is a value in a set *C* (categories).
	1. ***k*-nearest neighbors (KNN):** predicts the label of a data point by looking at the *k* closest labeled data points;
	
	* [scikit-learn example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_iris.py)

2. **Regression:** output is a value in R (continuous);
	* [scikit-learn example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Regression/scikitlearn_boston.py)

In Python we can use scikit-learn, TensorFlow and Keras for Supervised Learning.

## Unsupervised Learning
It tries to discover hidden patterns in unlabeled data. Main tasks:
1. **Clustering**: tries to discover the underlying groups (clusters) in a dataset;
	1. ***k*-means clustering:** finds clusters given a number of clusters (scikit-learn);
2. **Data visualization:** hierarchical clustering and *t*-SNE;
3. **Dimension reduction techniques:** Principal Component Analysis (PCA);
4. **Dimension reduction techniques:** Non-negative matrix factorization" ("NMF").

## Reinforcement Learning
Interaction with environment to learn how to optimize behavior, using a system of rewards and punishments.

**Useful links:**
1. [scikit-learn examples](https://scikit-learn.org/stable/auto_examples/index.html).
2. [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).
3. [Keras](https://keras.io/).
4. [Deep Learning Online Courses (free)](https://www.fast.ai/).
