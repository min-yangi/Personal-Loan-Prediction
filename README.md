# Personal-Loan-Prediction
This project predicts personal loan acceptance using the Kaggle 'Bank Personal Loan Modelling' dataset. Various ML models (SVM, KNN, Naive Bayes, Decision Tree, Logistic Regression) are compared with Neural Networks. Decision Tree achieves the highest accuracy.

## General Packages
import pandas as pd
import numpy as np
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

## Visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Datetime packages
from datetime import datetime
import time

## Preprocessing packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from statsmodels.formula.api import ols
import scipy.stats as stats ## Chi-quare


## SVM
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics

## Decision Tree
from sklearn.tree import DecisionTreeClassifier, export_text,plot_tree
from sklearn import tree

## Logistic Regression
from dmba import stepwise_selection
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import statsmodels.api as sm
from mord import LogisticIT
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

## KNN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import preprocessing

## Neural Network
from sklearn.neural_network import MLPClassifier,MLPRegressor

## Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, confusion_matrix
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

### Read Files
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv').rename({'Personal Loan':'Personal_Loan'},axis=1)
df.head(6)

## Checking data structure
df.shape

## Checking missing value and data types
df.info()

## Drop unnecessary columns
df.drop(['ID','ZIP Code'],axis='columns', inplace=True)

# Pre-processing

## Drop duplicates
df = df.drop_duplicates()
df.shape

## Separate numeric and categorical columns
num_col=['Age','Experience','Income','CCAvg','Mortgage']
cat_col=['Family','Education','Personal Loan','Securities Account','CD Account','Online','CreditCard']

# Create a function to check the dataframe for columns, data types, unique values, and null values

def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['column','dtypes','nunique','sum_null']
    return df_check
# Call the check function on the dataframe to return a new dataframe containing column information
check(df)

## Visualize Median price of each category
# matplotlib Version
plt.rcParams["figure.figsize"] = (35, 15)
sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows = 2, ncols = 3)
sns.barplot(x=df[["Family",'CreditCard']].value_counts().to_frame().reset_index()['Family'], y=df[["Family",'CreditCard']].value_counts().to_frame().reset_index()['count'],hue=df[["Family",'CreditCard']].value_counts().to_frame().reset_index()['CreditCard'],ax=axes[0,0])
sns.barplot(x=df[["Education",'CreditCard']].value_counts().to_frame().reset_index()['Education'], y=df[["Education",'CreditCard']].value_counts().to_frame().reset_index()['count'],hue=df[["Education",'CreditCard']].value_counts().to_frame().reset_index()['CreditCard'],ax=axes[0,1])
sns.barplot(x=df[["Personal_Loan",'CreditCard']].value_counts().to_frame().reset_index()['Personal_Loan'], y=df[["Personal_Loan",'CreditCard']].value_counts().to_frame().reset_index()['count'],hue=df[["Personal_Loan",'CreditCard']].value_counts().to_frame().reset_index()['CreditCard'],ax=axes[0,2])
sns.barplot(x=df[["Securities Account",'CreditCard']].value_counts().to_frame().reset_index()['Securities Account'], y=df[["Securities Account",'CreditCard']].value_counts().to_frame().reset_index()['count'],hue=df[["Securities Account",'CreditCard']].value_counts().to_frame().reset_index()['CreditCard'],ax=axes[1,0])
sns.barplot(x=df[["Online",'CreditCard']].value_counts().to_frame().reset_index()['Online'], y=df[["Online",'CreditCard']].value_counts().to_frame().reset_index()['count'],hue=df[["Online",'CreditCard']].value_counts().to_frame().reset_index()['CreditCard'],ax=axes[1,1])

plt.suptitle('') # Suppress the overall title
plt.tight_layout() #Increase the separation between the plots

# Continuous Features

### Checkign data distribution
df[['Experience','Income','CCAvg','Mortgage']].describe()

### Checkign data distribution
df[['Experience','Income','CCAvg','Mortgage']].describe()

## Visualize data distribution
# matplotlib Version
plt.rcParams["figure.figsize"] = (18, 10)
sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows = 2, ncols = 3)
sns.histplot(df['Age'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,0])
sns.histplot(df['Experience'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,1])
sns.histplot(df['Income'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,2])
sns.histplot(df['CCAvg'], color ='red', edgecolor='blue', bins = 30,ax=axes[1,0])
sns.histplot(df['Mortgage'], color ='red', edgecolor='blue', bins = 30,ax=axes[1,1])

plt.suptitle('') # Suppress the overall title
plt.tight_layout() #Increase the separation between the plot

# Correlation between continuous data
plt.figure(figsize=(10,8))
sns.heatmap(df[num_col].corr().round(2), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

### Standardlize data
df_normalize = df[['Age','Experience','Income','CCAvg']]
sc = StandardScaler()
sc.fit(df_normalize)
scaled_data = sc.transform(df_normalize)

# Applying PCA function
pca = PCA(n_components=2) ### we are trying to keep at least 95% of the variance of the original variables
x_pca = pca.fit_transform(scaled_data) ## Fit and transform the data
x_pca.shape ## Checking data dimensions

## Print out the variance ratio
explained_variance = pca.explained_variance_ratio_
explained_variance

### Create Data Frame for the new dimensions datas
df_final = pd.DataFrame(x_pca)
df_final.columns = ['PCA1','PCA2']
df_final.rename({'PCA1':'Age_Experience','PCA2':'Spending_on_income'})

## Visualize data distribution
# matplotlib Version
plt.rcParams["figure.figsize"] = (18, 10)
sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows = 2, ncols = 3)
sns.histplot(df['Age'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,0])
sns.histplot(df['Experience'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,1])
sns.histplot(df['Income'], color ='red', edgecolor='blue', bins = 30,ax=axes[0,2])
sns.histplot(df['CCAvg'], color ='red', edgecolor='blue', bins = 30,ax=axes[1,0])
sns.histplot(df['Mortgage'], color ='red', edgecolor='blue', bins = 30,ax=axes[1,1])

plt.suptitle('') # Suppress the overall title
plt.tight_layout() #Increase the separation between the plot

### Data finalized for models below
df_after_preprocessing = pd.merge(df,df_final,left_index=True, right_index=True,how='inner')
# df_after_preprocessing.to_csv('data_preprocessing.csv')

