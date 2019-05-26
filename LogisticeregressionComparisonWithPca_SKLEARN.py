#!/usr/bin/env python
# coding: utf-8

# In[27]:


from sklearn import datasets,decomposition,linear_model
import numpy as np
import time as time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[28]:


data=datasets.load_breast_cancer()
X=data.data
print(X.shape)


# In[29]:


sc=StandardScaler()
X_std=sc.fit_transform(X)
xtrain,xtest,ytrain,ytest=train_test_split(X_std,data.target,random_state=0)


# In[30]:


pca=decomposition.PCA(n_components=15)
xtrain_pca=pca.fit_transform(xtrain)
xtest_pca=pca.transform(xtest)


# In[31]:


lr=linear_model.LogisticRegression()
start=time.time()
lr.fit(xtrain,ytrain)
end=time.time()
print(end-start)
print(lr.score(xtest,ytest))


# In[32]:


lr=linear_model.LogisticRegression()
start=time.time()
lr.fit(xtrain_pca,ytrain)
end=time.time()
print(end-start)
print(lr.score(xtest_pca,ytest))


# In[ ]:




