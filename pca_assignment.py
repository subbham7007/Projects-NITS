#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import math


# In[2]:


df = pd.read_csv("Iris/Iris.csv")
df.shape


# In[3]:


df.head()


# In[4]:


df.set_index('Id', inplace = True)
df.head()


# In[5]:


df['SLength_mean'] = df.SepalLengthCm.mean().round(3)
df['SWidth_mean'] = df.SepalWidthCm.mean().round(3)
df['PLength_mean'] = df.PetalLengthCm.mean().round(3)
df['PWidth_mean'] = df.PetalWidthCm.mean().round(3)
df.head()


# In[6]:


df['SL-SL_mean'] = (df['SepalLengthCm'] - df['SLength_mean']).round(3)
df['SW-SW_mean'] = (df['SepalWidthCm'] - df['SWidth_mean']).round(3)
df['PL-PL_mean'] = (df['PetalLengthCm'] - df['PLength_mean']).round(3)
df['PW-PW_mean'] = (df['PetalWidthCm'] - df['PWidth_mean']).round(3)
df.head()


# In[ ]:





# In[7]:


class pca:
    def __init__ (self, df):
        self.pca = None
        self.sub_mean_vec = np.array(df)
        self.no_records = self.sub_mean_vec.shape[0]
        self.no_features = self.sub_mean_vec.shape[1]
        self.covar_matrix_add = [[0 for _ in range(self.no_features)] for _ in range(self.no_features)]
        self.covar_matrix = self.covar_matrix_add
        # self.covar_matrix = self.get_covar_matrix()
    
    def get_covar_matrix(self):
        
        for vi in self.sub_mean_vec:
                x_arr = np.array(vi).reshape((len(vi)), 1)
                xt_arr = np.array(vi).reshape(1, (len(vi)))
                
                matrix_mul = np.dot(x_arr,xt_arr).round(3)
                
                self.covar_matrix_add = np.add(self.covar_matrix_add, matrix_mul)
        self.covar_matrix = self.covar_matrix_add / self.no_records
        return self.covar_matrix
    
    def get_pca(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covar_matrix)
        self.pca = self.eigenvectors[:,np.argmax(self.eigenvalues)]
        return self.pca


# In[8]:


co_var_df = df[['SL-SL_mean', 'SW-SW_mean', 'PL-PL_mean', 'PW-PW_mean']]


# In[9]:


p = pca(co_var_df)
p.no_records, p.no_features


# In[10]:


p.get_covar_matrix()


# In[11]:


p.covar_matrix_add


# In[12]:


p.get_pca()


# In[13]:


p.eigenvalues, p.eigenvectors


# In[ ]:




