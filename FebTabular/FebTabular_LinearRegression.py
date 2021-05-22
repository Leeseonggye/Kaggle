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


# ## Data preprocess

# In[2]:


df = pd.read_csv("FebTabulartrain.csv")


# In[3]:


df


# In[4]:


df.info()


# In[8]:


def load_file(path):
    data = pd.read_csv(path)
    for i in range(0,10):        
        data.loc[data["cat%d" %i] == "A", "cat%d" %i] =1
        data.loc[data["cat%d" %i] == "B", "cat%d" %i] =2
        data.loc[data["cat%d" %i] == "C", "cat%d" %i] =3
        data.loc[data["cat%d" %i] == "D", "cat%d" %i] =4
        data.loc[data["cat%d" %i] == "E", "cat%d" %i] =5
        data.loc[data["cat%d" %i] == "F", "cat%d" %i] =6
        data.loc[data["cat%d" %i] == "G", "cat%d" %i] =7
        data.loc[data["cat%d" %i] == "H", "cat%d" %i] =8
        data.loc[data["cat%d" %i] == "I", "cat%d" %i] =9
        data.loc[data["cat%d" %i] == "J", "cat%d" %i] =10
        data.loc[data["cat%d" %i] == "K", "cat%d" %i] =11
        data.loc[data["cat%d" %i] == "L", "cat%d" %i] =12
        data.loc[data["cat%d" %i] == "M", "cat%d" %i] =13
        data.loc[data["cat%d" %i] == "N", "cat%d" %i] =14
        data.loc[data["cat%d" %i] == "O", "cat%d" %i] =15


             
    return data


# In[10]:


df_preprocess = load_file("FebTabulartrain.csv")


# In[18]:


df_preprocess.keys()


# ## Modeling
# ### Linear Regression

# In[24]:


PREDICTORS = ["cat%d" %i for i in range(0,10)] + ["cont%d" %i for i in range(0,14)] 
TARGET = ["target"]


# In[28]:


def apply_linear_regression(train,test):
    alg = LinearRegression()
    alg.fit(train[PREDICTORS],train[TARGET])
    print(alg.intercept_,alg.coef_)
    
    predictions = alg.predict(test[PREDICTORS])
        
    output = pd.DataFrame({
        "id": test["id"],
        "target" : predictions[:,0].astype(float)
    })
    output.to_csv("FebTabular_submission.csv",index = False)


# ## Prediction

# In[29]:


train = load_file("FebTabulartrain.csv")
test = load_file("FebTabulartest.csv")


# In[30]:


apply_linear_regression(train,test)

