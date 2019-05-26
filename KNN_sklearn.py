#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from collections import Counter


# In[5]:


dataset=datasets.load_breast_cancer()
# dataset.target
# dataset.data


# In[6]:


X_train,X_test,Y_train,Y_test=train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=0)


# In[7]:


clf=KNeighborsClassifier(n_neighbors=7)


# In[8]:


clf.fit(X_train,Y_train)


# In[9]:


clf.score(X_test,Y_test)


# In[11]:


x_axis=[]
y_axis=[]
for i in range(1,26,2):
    clf=KNeighborsClassifier(n_neighbors = i)
    x_axis.append(i)
    score=cross_val_score(clf,X_train,Y_train,cv=3)
    y_axis.append(score.mean())
#     print(i,score.mean())
import matplotlib.pyplot as plt
plt.plot(x_axis,y_axis)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




