# Machine Learning Algorithms

### Conventions:
* features = predictor variables = independent variables;
* target variable = dependent variable = response variable;
* training a model on the data = fitting a model to the data.

## Supervised Learning
It uses labeled data as input, which are usually represented in a table structure. It divides in:
1. **Classification:** outputs a label, which is a value in a set *C* (categories).
	* **Confusion matrix:** table used to evaluate the performance of a classification model.
	1. ***k*-nearest neighbors (KNN):** predicts the label of a data point by looking at the *k* closest labeled data points. **Code example:** [sklearn iris species example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_iris.py)
	2. **Logistic Regression:** outputs probabilities.
		* One of the most commonly used ML algorithms for two-class classification;
		* Dependent variable follows Bernoulli Distribution;
		* Logistic regression threshold is 0.5, by default.
		* **ROC curve (Receiver Operating Characteristic Curve):** series of different false positive and true positive rates as we vary threshold;
		* **Estimation:** Maximum Likelihood Estimation (MLE);
		* Model fitness calculated through Concordance and KS-Statistics;
		* It divides in:
			1. **Binary Logistic Regression:** target variable has two possible outputs. **Examples:** spam detection, diabetes prediction, cancer detection, and if a user will click on an advertisement link or buy a product or not. **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)
			2. **Multinomial Logistic Regression:** target variable has 3 or more **nominal** categories. **Examples:** types of iris flowers, and types of wine. **Code example:** [sklearn iris species example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_iris2.py)
			3. **Ordinal Logistic Regression:** target variable has 3 or more **ordinal** categories. **Example:** restaurant or product rating (from 1 to 5), and classification of documents into categories.
		
	3. **Support Vector Machines (SVM)**
		* **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)
	4. **Kernel SVM**
		* **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)	
	5. **Naïve Bayes**
		* **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)	
	6. **Decision Tree Algorithm**
		* **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)	
	7. **Random Forest Classification**
		* **Code example:** [sklearn breast cancer example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Classification/scikitlearn_breast_cancer.py)	

2. **Linear Regression:** output is a value in R (continuous);
	* **Estimation:** OLS (Ordinary Least Squares), which is the sum of the squares of the residuals (same as minimizing the mean squared error of the predictions on the training set);
	* **Code example:** [sklearn Boston housing prices example](https://github.com/letyrobueno/Machine-Learning/blob/master/Supervised-Learning/Regression/scikitlearn_boston.py)

[Linear Regression X Logistic Regression:](https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python)
	* Linear regression gives a continuous output (example: house price), logistic regression gives a constant output (example: stock price);
	* Linear regression estimated by Ordinary Least Squares (OLS), logistic regression estimated by Maximum Likelihood Estimation (MLE);

In Python we can use scikit-learn (sklearn), TensorFlow and Keras for Supervised Learning.

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
