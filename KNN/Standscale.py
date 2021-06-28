
# coding: utf-8

# In[1]:

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


# In[2]:

data = load_iris().data


# In[3]:

p_data = pd.DataFrame(data,columns=['sepal length (cm)',
  'sepal width (cm)',
  'petal length (cm)',
  'petal width (cm)'])


# In[4]:

def standscale(p_data):
    data_min = p_data.min()
    data_max = p_data.max()
    scale_data = (p_data - data_min)/(data_max - data_min)
    return scale_data


# In[5]:

s_data = standscale(p_data)
s_data



