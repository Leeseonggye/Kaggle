#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import linear_model


# In[63]:


data = pd.read_csv("train.csv")
data.info()
data.isna().sum()


# ## Data preprocess #1
# ### Sex -> (0,1) coding
# ### Embarked -> fill na median

# In[64]:


def load_file(path):
    data = pd.read_csv(path)
    data["Sex"] = data["Sex"].apply(lambda sex: 1 if sex == "male" else 0)
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] =0
    data.loc[data["Embarked"] == "C", "Embarked"] =1
    data.loc[data["Embarked"] == "Q", "Embarked"] =2
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
             
    return data


# In[65]:


train_raw = load_file("train.csv")
test_raw = load_file("test.csv")
test_raw["Age"][0:2].isna()


# In[66]:


test_raw["Age"][0:2].isna()


# ## Data preprocess #2
# ### Age -> Predict missing value with linear regression (Linear imputation)

# In[67]:


PREDICTORS_AGE = ["Pclass","Sex","SibSp","Parch","Fare","Embarked"]
TARGET_AGE = ["Age"]


# In[68]:


def Age_Linear_Regression(train, test):
    alg = LinearRegression()
    alg.fit(train[PREDICTORS_AGE],train[TARGET_AGE])
    predictions = alg.predict(test[PREDICTORS_AGE])
    for age in range(len(test["Age"])):
        if test["Age"].isna()[age] == True:
            test["Age"][age] = predictions[age]


# In[69]:


train_age = train_raw.dropna(subset = ["Age"])


# In[70]:


Age_Linear_Regression(train_age, train_raw)


# In[58]:


train_age = train_raw.dropna(subset = ["Age"])
test_age = test_raw


# In[59]:


Age_Linear_Regression(train_age, test_age)


# In[60]:


test_age["Age"].describe()


# ### Age < 0 -> Age = 0.5 (영유아로 가정)

# In[72]:


def Age_revised(age):
    if age >=0:
        return age
    else:
        return 0.5


# In[73]:


train_raw.Age = train_raw.Age.apply(Age_revised)
test_age.Age = test_age.Age.apply(Age_revised)


# ## Modeling

# In[74]:


PREDICTORS = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
TARGET = ["Survived"]


# In[79]:


def apply_linear_regression(train,test):
    alg = LinearRegression()
    alg.fit(train[PREDICTORS],train[TARGET])
    print(alg.intercept_,alg.coef_)
    
    predictions = alg.predict(test[PREDICTORS])
    predictions[predictions<=0.5] = 0
    predictions[predictions>0.5] = 1
    
    output = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived" : predictions[:,0].astype(int)
    })
    output.to_csv("logistic_regression_age_revised.csv",index = False)


# In[80]:


apply_linear_regression(train_raw,test_age)

