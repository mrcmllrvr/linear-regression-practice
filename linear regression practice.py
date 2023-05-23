#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set() #for fancy graphs


# In[2]:


data = pd.read_csv(r"...Simple linear regression.csv")
data


# In[3]:


data.describe()


# In[4]:


y = data["GPA"]
x1 = data["SAT"]


# In[5]:


plt.scatter(x1,y)
plt.xlabel("SAT", fontsize = 15)
plt.ylabel("GPA", fontsize = 15)
plt.show()


# In[6]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[7]:


plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange', label = 'regression line')
plt.xlabel("SAT", fontsize = 15)
plt.ylabel("GPA", fontsize = 15)
plt.show()


# ## Making predictions with Linear Regression (SAT and GPA scores)

# In[8]:


raw_pred_data = pd.read_csv(r"C:\Users\maria\OneDrive\Documents\Camille\BAITO\PARA SA KINABUKASAN\The Data Science Course 2021 - All Resources\Part_5_Advanced_Statistical_Methods_(Machine_Learning)\S33_L203\1.03. Dummies.csv")


# In[9]:


pred_data = raw_pred_data.copy()
pred_data


# In[10]:


pred_data["Attendance"] = pred_data["Attendance"].map({"Yes": 1, "No": 0})
pred_data


# In[11]:


y_pred = pred_data["GPA"]
x1_pred = pred_data[["SAT", "Attendance"]]


# In[12]:


x_pred = sm.add_constant(x1_pred)
results_pred = sm.OLS(y_pred, x_pred).fit()
results_pred.summary()


# ## Multiple Regression dealing with dummy data

# In[13]:


raw_multiple_reg = pd.read_csv(r"C:\Users\maria\OneDrive\Documents\Camille\BAITO\PARA SA KINABUKASAN\The Data Science Course 2021 - All Resources\Part_5_Advanced_Statistical_Methods_(Machine_Learning)\S33_L204\real_estate_price_size_year_view.csv")


# In[14]:


raw_multiple_reg.head()


# In[15]:


#just copied to another variable for cleaning
multiple_reg = raw_multiple_reg.copy() 


# In[16]:


#changing values of column view to 1 or 0
multiple_reg["view"] = multiple_reg["view"].map({"Sea view": 1, "No sea view": 0})
multiple_reg


# In[17]:


#declaring dependent and independent variables

y_mul = multiple_reg["price"]
x1_mul = multiple_reg[["size","year","view"]]


# In[18]:


#regression

x_mul = sm.add_constant(x1_mul)
results_mul = sm.OLS(y_mul, x_mul).fit()
results_mul.summary()

