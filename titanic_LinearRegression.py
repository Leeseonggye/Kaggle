#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[22]:


PREDICTORS = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
TARGET = ["Survived"]


# In[23]:


def load_file(path):
    data = pd.read_csv(path)
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Sex"] = data["Sex"].apply(lambda sex: 1 if sex == "male" else 0)
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] =0
    data.loc[data["Embarked"] == "C", "Embarked"] =1
    data.loc[data["Embarked"] == "Q", "Embarked"] =2
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
             
    return data


# ## Linear Regression

# In[24]:


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
    output.to_csv("logistic_regression.csv",index = False)


# In[26]:


train = load_file("train.csv")
test = load_file("test.csv")


# In[27]:


train


# In[29]:


apply_linear_regression(train,test)

