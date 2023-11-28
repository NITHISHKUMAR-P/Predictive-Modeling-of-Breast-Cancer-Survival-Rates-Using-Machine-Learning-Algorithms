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
df.drop(columns=["Unnamed: 32","id"], axis=1, inplace=True)
num_cols = df.columns
num_cols

# Exploratory Data Analysis
fig,axes = plt.subplots(nrows = 10, ncols = 3, figsize = (20, 40))
axes = axes.flat
data=df
color_cycle = list(plt.rcParams['axes.prop_cycle'])
num_colors = len(color_cycle)

for i, num_col in enumerate(num_cols):
    sns.histplot(data,
               x = num_col,
               stat = 'count',
               kde = True,
               color = color_cycle[i % num_colors]["color"],
               line_kws = {'linewidth': 2,
                           'linestyle':'dashed'},
               alpha = 0.4,
               ax = axes[i])
    sns.rugplot(data,
              x = num_col,
              color = color_cycle[i % num_colors]["color"],
              ax = axes[i], alpha = 0.7)
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Count", fontsize = 7, fontweight = 'bold', color = 'black')
    axes[i].set_title(num_col, fontsize = 8, fontweight = 'bold', color = 'black')
    axes[i].tick_params(labelsize = 6)

fig.delaxes(axes[5])
fig.suptitle('Distribution of numerical variables', fontsize = 12, fontweight = 'bold', color = 'darkred', y = 0.92)
fig.tight_layout()
fig.subplots_adjust(top = 0.9)
fig.show()

corr_matrix = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig,ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr_matrix,
            mask = mask,
            square = True,
            annot = True,
            ax = ax,
            linewidths = 0.2,
            annot_kws = {'size':6, 'fontweight':'bold'},
            cmap='coolwarm')
ax.tick_params(labelsize=8, color = 'blue', labelcolor='black')
ax.set_title('Correlation of numerical variables',fontsize = 15,fontweight='bold', color = 'darkblue')
fig.show()

# ML algorithms
def plot_metrics(cv_results,model_name):
    metrics = ['accuracy', 'precision','recall', 'f1']

    metrics_train = {'accuracy':round(cv_results['train_accuracy'].mean(), 3),
                'precision':round(cv_results['train_precision'].mean(), 3),
                'recall':round(cv_results['train_recall'].mean(), 3),
                'f1':round(cv_results['train_f1'].mean(), 3)}

    metrics_test = {'accuracy':round(cv_results['test_accuracy'].mean(), 3),
                    'precision':round(cv_results['test_precision'].mean(), 3),
                    'recall':round(cv_results['test_recall'].mean(), 3),
                    'f1':round(cv_results['test_f1'].mean(), 3)}

    df_metrics = pd.DataFrame(index = metrics,
                              data = {'Train':[metrics_train[metric] for metric in metrics],
                                      'Test':[metrics_test[metric] for metric in metrics]})


    n = len(df_metrics.index)
    x = np.arange(n)


    width = 0.25

    fig,ax = plt.subplots(figsize=(6,4))

    rects1 = ax.bar(x-width, df_metrics.Train, width=width, label='Train',linewidth=1.6,edgecolor='black',color='blue')

    rects2 = ax.bar(x, df_metrics.Test, width=width, label='Test',linewidth=1.6, edgecolor='black', color = 'red')

    ax.set_title(f'Metrics of {model_name}',fontsize=12, fontweight='bold')
    ax.set_ylabel('Score',fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x-0.13)
    ax.set_xticklabels(df_metrics.index, fontsize=10, fontweight='bold')
    ax.legend()

    def autolabel(rects):

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height-0.005),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom', size = 7, weight = 'bold')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    fig.show()

def plot_accuracy(cv_results,model_name):
    fig,axes = plt.subplots()
    axes.plot(np.arange(1,6,1),cv_results['train_accuracy'], '-o', linestyle = 'dashed', label = 'Train')
    axes.plot(np.arange(1,6,1),cv_results['test_accuracy'], '-o', linestyle = 'dashed', label = 'Test')
    axes.set_xticks(np.arange(1,6,1))
    axes.set_xlabel('CV', fontsize = 10, fontweight = 'bold', color = 'black')
    axes.set_ylabel('Score', fontsize = 10, fontweight = 'bold', color = 'black')
    axes.set_title(f'Accuracy of {model_name}', fontsize = 12, fontweight = 'bold', color = 'blue')
    axes.legend()

    fig.tight_layout()
    fig.subplots_adjust(top = 0.9)
    fig.show()

X=df.drop("diagnosis", axis=1)
y=df.diagnosis

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

skf = StratifiedKFold(n_splits = 5,
                      shuffle = True,
                      random_state = 123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
best_trial_params={'max_depth': 11, 'subsample': 0.6, 'n_estimators': 6200, 'eta': 0.08,
                   'reg_alpha': 1, 'reg_lambda': 21, 'min_child_weight': 12,
                   'colsample_bytree': 0.15507995688972184}

XGB_tuned = XGBClassifier(**best_trial_params)

from sklearn.metrics import accuracy_score

models = [LogisticRegression(max_iter=2000) ,RandomForestClassifier(), SVC(),  KNeighborsClassifier(), GaussianNB(), SGDClassifier(),XGBClassifier(), CatBoostClassifier(verbose=False),XGB_tuned]
models_cv_results_train,models_cv_results_test = {},{}
names = ['Logistic Regression','Random Forest','SVM','Gaussin','SGD','XGB','Cat Boost','XGB Tuned']

for model,name in zip(models,names):
    cv_results = cross_validate(estimator = model,
                               X = X,
                               y = y,
                               scoring = ['accuracy', 'precision', 'recall', 'f1'],
                               cv = skf,
                               verbose = 1,
                               return_train_score = True,
                               error_score = 'raise')
    plot_accuracy(cv_results,name)
    plot_metrics(cv_results,name)
    model.fit(X_train,y_train)
    y_predss=model.predict(X_test)
    cm = confusion_matrix(y_test, y_predss)
    plt.figure(figsize=(10, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='BuPu', linewidths = 2, linecolor = "black", square=True, cbar=True,
        xticklabels=["B", "M"],
        yticklabels=["B", "M"]
    )

    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.title(f'Confusion Matrix for {name}', fontsize=16, fontweight='bold')
    plt.yticks(rotation=360)

    plt.show()
    metrics_train= {'accuracy':round(cv_results['train_accuracy'].mean(), 3),
                'precision':round(cv_results['train_precision'].mean(), 3),
                'recall':round(cv_results['train_recall'].mean(), 3),
                'f1':round(cv_results['train_f1'].mean(), 3)}
    metrics_test= {'accuracy':round(cv_results['test_accuracy'].mean(), 3),
                'precision':round(cv_results['test_precision'].mean(), 3),
                'recall':round(cv_results['test_recall'].mean(), 3),
                'f1':round(cv_results['test_f1'].mean(), 3)}
    models_cv_results_train[name] = metrics_train
    models_cv_results_test[name] = metrics_test
    print("--------------------------------------------------------------------------")

models=[X]

df_train_metrics = pd.DataFrame.from_dict(models_cv_results_train,orient='index')
# We order from highest to lowest by the f1 score metric, which is the one we chose because we have unbalanced classes.
df_train_metrics = df_train_metrics.sort_values('recall', ascending = False)

# we visualize the training metrics
fig,ax = plt.subplots(figsize=(9,4.5))
sns.heatmap(df_train_metrics, annot=True, cmap = 'coolwarm', annot_kws = {'fontweight':'bold'},fmt = '.3f', ax = ax)
ax.xaxis.tick_top()
ax.set_ylabel('Models', fontsize = 11, fontweight = 'bold', color = 'blue')
ax.set_title('Metrics train', fontsize = 11, fontweight = 'bold', color = 'blue')
fig.show()

df_train_metrics = pd.DataFrame.from_dict(models_cv_results_test,orient='index')
# We order from highest to lowest by the f1 score metric, which is the one we chose because we have unbalanced classes.
df_train_metrics = df_train_metrics.sort_values('recall', ascending = False)

# we visualize the training metrics
fig,ax = plt.subplots(figsize=(9,4.5))
sns.heatmap(df_train_metrics, annot=True, cmap = 'coolwarm', annot_kws = {'fontweight':'bold'},fmt = '.3f', ax = ax)
ax.xaxis.tick_top()
ax.set_ylabel('Models', fontsize = 11, fontweight = 'bold', color = 'blue')
ax.set_title('Metrics test', fontsize = 11, fontweight = 'bold', color = 'blue')
fig.show()
```
## Output:
### Heat map:
![image](https://github.com/NITHISHKUMAR-P/Predictive-Modeling-of-Breast-Cancer-Survival-Rates-Using-Machine-Learning-Algorithms/assets/93427017/2d6ae74c-1c33-4f30-b151-539f7d54709c)
### EDA:
![image](https://github.com/NITHISHKUMAR-P/Predictive-Modeling-of-Breast-Cancer-Survival-Rates-Using-Machine-Learning-Algorithms/assets/93427017/a17d8722-462c-46cb-8694-e6dcbdd8c422)
### Correlation:
![image](https://github.com/NITHISHKUMAR-P/Predictive-Modeling-of-Breast-Cancer-Survival-Rates-Using-Machine-Learning-Algorithms/assets/93427017/ac9dce41-3721-4955-9d15-04e2f95d2415)
### 

