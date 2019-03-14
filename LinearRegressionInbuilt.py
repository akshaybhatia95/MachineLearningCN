#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[6]:


from sklearn import datasets


# In[33]:


boston=datasets.load_boston()
print(boston)
X=boston.data
Y=boston.target
print(X.shape)
print(Y.shape)


# In[31]:


import pandas as pd
df=pd.DataFrame(X)
print(boston.feature_names)
df.columns=boston.feature_names
print(df.columns)
df.describe()


# In[35]:


boston.DESCR


# In[37]:


from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,Y)


# In[ ]:





# In[45]:


from sklearn.linear_model import LinearRegression
algo1=LinearRegression()
algo1.fit(x_train,y_train)
y_pred=algo1.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(y_pred,y_test)
plt.show()

