#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


titanic_data =pd.read_csv('Titanic.csv')


# In[3]:


titanic_data.head(10)


# In[4]:


sns.countplot(x="Survived", data=titanic_data)


# In[5]:


sns.countplot(x="Survived", hue="Sex", data=titanic_data)


# In[6]:


sns.countplot(x="Survived", hue="Pclass", data=titanic_data)


# In[7]:


titanic_data.isnull()


# In[8]:


titanic_data.isnull().sum()


# In[9]:


sns.boxplot(x="Pclass", y="Age", data=titanic_data)


# In[10]:


titanic_data.head(5)


# In[11]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[12]:


titanic_data.head(5)


# In[13]:


titanic_data.dropna(inplace= True)


# In[14]:


titanic_data.isnull().sum()


# In[15]:


Sex = pd.get_dummies(titanic_data['Sex'],drop_first= True)


# In[16]:


Embark = pd.get_dummies(titanic_data['Embarked'],drop_first= True)


# In[17]:


Pclass = pd.get_dummies(titanic_data['Pclass'],drop_first= True)


# In[18]:


titanic_data= pd.concat([titanic_data,Sex,Embark,Pclass],axis=1)


# In[19]:


titanic_data.head(10)


# In[20]:


titanic_data.drop(['Sex','Name','Pclass','PassengerId','Ticket','Embarked'], axis= 1, inplace= True)


# In[21]:


titanic_data.head(10)


# In[22]:


X = titanic_data.drop("Survived",axis = 1)
y= titanic_data["Survived"]


# In[25]:


import sklearn


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=1)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


logmodel= LogisticRegression()


# In[33]:


logmodel.fit(X_train,y_train)


# In[34]:


predictions=logmodel.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report


# In[36]:


classification_report(y_test,predictions)


# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


accuracy_score(y_test,predictions)


# In[ ]:




