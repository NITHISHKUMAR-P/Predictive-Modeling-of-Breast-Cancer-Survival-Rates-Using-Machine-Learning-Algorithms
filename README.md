# Predictive-Modeling-of-Breast-Cancer-Survival-Rates-Using-Machine-Learning-Algorithms
In this project we employed eight machine learning algorithms, including XGBClassifier, CatBoostClassifier, LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier, GaussianNB, and SGDClassifier, to predict breast cancer survival rates using the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset comprises 569 instances with 32 attributes, and classifies cases as malignant or benign.
## About:
This project aims to contribute to refined breast cancer prognosis by identifying the most effective algorithm for accurate classification. The results have implications for clinicians and researchers, offering insights into the potential integration of advanced machine learning tools in oncology decision-making. The project's systematic approach and robust dataset underscore the significance of this research in advancing our understanding of breast cancer survival prediction.
## Project Flow:
![image](https://github.com/NITHISHKUMAR-P/Predictive-Modeling-of-Breast-Cancer-Survival-Rates-Using-Machine-Learning-Algorithms/assets/93427017/2e1ea4b8-75d1-4508-8a7a-8e85f324bc24)

## Training Model:
### XGBClassifier:
It refers to Extreme Gradient Boosting classifier, which is a popular machine learning algorithm for classification tasks. XGBoost is an efficient and scalable implementation of gradient boosting. 
### CatBoostClassifier:
It refers to the classifier implementation in the CatBoost library, which is another popular gradient boosting library for machine learning. CatBoost is designed to handle categorical features efficiently without requiring explicit pre-processing like one-hot encoding. 
### LogisticRegression:
It is a linear model for binary classification that predicts the probability of an instance belonging to a particular class. It is used for binary classification problems, not regression problems.
### RandomForestClassifier:
It is an ensemble learning method for classification that operates by constructing a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
### SVC:
It stands for Support Vector Classification, and it's a type of Support Vector Machine (SVM) algorithm used for classification tasks. SVMs are supervised learning models that analyze data and recognize patterns, with applications in both classification and regression analysis.
### KNeighborsClassifier:
It is a simple and widely used algorithm for classification in machine learning. It's a type of instance-based learning or lazy learning where the model makes predictions based on the majority class of the k-nearest neighbors of a data point. In other words, it classifies a data point based on the classes of its nearest neighbors in the feature space.
### GaussianNB:
It stands for Gaussian Naive Bayes, and it is a variant of the Naive Bayes algorithm for classification. Naive Bayes is a probabilistic algorithm based on Bayes' theorem, and it assumes that the features used to describe an observation are conditionally independent given the class label. 
### SGDClassifier:
It stands for Stochastic Gradient Descent Classifier, and it is a linear model trained using stochastic gradient descent (SGD) optimization. It is part of scikit-learn's 'linear_model' module and can be used for binary and multiclass classification.
## Dataset:
The project utilizes the Breast Cancer Wisconsin (Diagnostic) dataset, comprising 569 instances with 32 attributes. The scope involves leveraging this dataset to enhance predictive accuracy and refine breast cancer prognosis.
## Results and Evaluation:
After training the model, we conducted an evaluation to assess its performance. We used various metrics to measure the accuracy, precision, recall, and F1-score of our object detection model.
## Python Code:
```py
# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import random

# Dataset
df = pd.read_csv('/content/drive/MyDrive/College/Mini Project/Nithish_Project/Colab Files/Breast_Cancer_Nipple.csv')
df
df.sample(5).T
df.info()  # clean data !
df.describe().T
df.isnull().sum()
df.duplicated().sum()

# Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
df.diagnosis.tail(10)

# Correlation
correlation_index=df.corrwith(df.diagnosis)
correlation_index.sort_values(ascending=False)
sns.heatmap(df.corr(), cmap='YlGnBu')

# Drop invalid columns

```
