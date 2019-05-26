#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn import svm, datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[33]:


iris=datasets.load_iris()
x=iris.data
y=iris.target


# In[34]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0)


# In[37]:


clf=svm.SVC(gamma="auto")
clf.fit(xtrain,ytrain)


# In[38]:


clf.score(xtest,ytest)


# In[43]:


import numpy as np
a=np.arange(1,3,0.2)
b=np.arange(4,6,0.2)
xx,yy=np.meshgrid(a,b)
yy.shape[0]


# In[ ]:




