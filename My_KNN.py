#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from collections import Counter

dataset=datasets.load_breast_cancer()
# dataset.target
# dataset.data
X_train,X_test,Y_train,Y_test=train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=0)


# In[7]:


def train(x,y):
    return
def predict_one(x_train,y_train,x_test,k):
    distances=[]
    for i in range(len(x_train)):
        distance = ((x_train[i,:]-x_test)**2).sum()
        distances.append([distance,i])
    distances=sorted(distances)
    targets=[]
    for i in range(k):
        targets.append(y_train[distances[i][1]])
    return Counter(targets).most_common()[0][0]
def predict(x_train,y_train,x_test_data,k):
    predictions=[]
    for x_test in x_test_data:
        predictions.append(predict_one(x_train,y_train,x_test,7))
    return predictions


# In[8]:


y_pred=predict(X_train,Y_train,X_test,7)
accuracy_score(y_pred,Y_test)       


# In[ ]:




