#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np


# In[14]:


def fit(X_train,Y_train):
    result={}
    class_values=set(Y_train)
    for curr_class in class_values:
        result[curr_class]={}
        result["total_data"]=len(Y_train)
        current_class_rows=(Y_train==curr_class)
        X_train_current=X_train[current_class_rows]
        Y_train_current=Y_train[current_class_rows]
        num_features=X_train.shape[1]
        result[curr_class]["total_count"]=len(Y_train_current)
        for j in range(1,num_features+1):
            result[curr_class][j]={}
            all_possible_values=X_train[:,j-1]
            for curr_value in all_possible_values:
                 result[curr_class][j][curr_value]=(X_train_current[:,j-1]==curr_value).sum()
    return result                


# In[15]:


def predict(dictionary,X_test):
    y_pred=[]
    for x in X_test:
        x_class=predictSinglePoint(dictionary,x)
        y_pred.append(x_class)
        
    return y_pred


# In[16]:


def predictSinglePoint(dictionary,x):
    classes=dictionary.keys()
    best_p=-1;
    best_class=-1000
    first_run=True
    for curr_class in classes:
        if(curr_class=="total_data"):
            continue
        p_curr_class=probability(dictionary,x,curr_class)
        if(first_run==True or p_curr_class > best_p):
            best_p=p_curr_class
            best_class=curr_class
            first_run=False
    return best_class
        


# In[17]:


def probability(dictionary,x,y):
    output=np.log(dictionary[y]["total_count"])-np.log(dictionary["total_data"])
    num_features=len(dictionary[y].keys())-1
    for j in range(1,num_features+1):
        xj=x[j-1]
        count_current_class_with_value_xj=dictionary[y][j][xj]+1
        count_current_class=dictionary[y]["total_count"]+len(dictionary[y][j].keys())
        curr_xj_probability=np.log(count_current_class_with_value_xj)-np.log(count_current_class)
        output=output*curr_xj_probability
    return output
    
    


# In[23]:


def makeLabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5*second_limit
    for i in range (0,len(column)):
        if (column[i] < first_limit):
            column[i] = 0
        elif (column[i] < second_limit):
            column[i] = 1
        elif(column[i] < third_limit):
            column[i] = 2
        else:
            column[i] = 3
    return column
from sklearn import datasets
iris = datasets.load_iris()
print(iris.feature_names)
print(iris)
X = iris.data
Y = iris.target
for i in range(0,X.shape[-1]):
    X[:,i] = makeLabelled(X[:,i])
from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.25,random_state=0)
dictionary = fit(X_train,Y_train)
Y_pred = predict(dictionary,X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




