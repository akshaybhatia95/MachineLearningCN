#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[3]:


np.random.seed(2343243)


# In[4]:


mean_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1 = np.random.multivariate_normal(mean_vec1, cov_mat1, 100)


# In[5]:


mean_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2 = np.random.multivariate_normal(mean_vec2, cov_mat2, 100)


# In[6]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
ax.plot(class1[:, 0], class1[:, 1], class1[:, 2], 'o')
ax.plot(class2[:, 0], class2[:, 1], class2[:, 2], '^')
plt.show()


# In[7]:


all_data = np.concatenate((class1, class2))


# In[8]:


pca = PCA(n_components = 2)
transformed_data = pca.fit_transform(all_data)
transformed_data


# In[9]:


pca.components_


# In[10]:


plt.plot(transformed_data[0:100,0],transformed_data[0:100,1],"o")
plt.plot(transformed_data[100:200,0],transformed_data[100:200,1],"^")
plt.show()


# In[11]:


X_approx = pca.inverse_transform(transformed_data)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
ax.plot(X_approx[:, 0], X_approx[:, 1], X_approx[:, 2], '^')
plt.show()


# In[12]:


a = -0.409689
b = 7.2827
c = - 7.1008
i = 10
a * X_approx[i][0] + b* X_approx[i][1] + c * X_approx[i][2]


# In[13]:


# own pca using numpy
all_data_t=all_data.T
cov=np.cov(all_data_t)
cov


# In[15]:


eig_vals,eig_vec=np.linalg.eig(cov)


# In[17]:


eig_val_vec_pair=[]
for i in range(len(eig_vals)):
    eig_v=eig_vec[:,i]
    eig_val_vec_pair.append((eig_vals[i],eig_v))
eig_val_vec_pair.sort(reverse=True)
print(eig_val_vec_pair)


# In[18]:


# inBuilt eigenvectors same as calculated
pca.components_ 


# In[19]:


# Inbuilt eigenvalues same as calcualted
pca.explained_variance_


# In[ ]:




