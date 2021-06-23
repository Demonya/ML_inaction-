

# coding: utf-8

# # KNN：
#  类别划分：有监督
#  
#  算法思想：通过计算预测数据与已知数据的欧式距离（亦可使用其他距离进行衡量），选出前K个最近距离的类别，作为预测数据的类别。
#  以鸢尾花为例：
#  测试数据集中有[5.9,3.0,4.2,1.5]的测试样本，计算该样本到测试数据集中某样本[6,3,4,1.5]欧式距离:
#      
#      d = sqrt{(5.9-6)^2+(3-3)^2+(4.2-2)^2+(1.5-1.5)^2}
#  
#  计算出预测样本到每个测试数据集的距离，选出前K个距离最小的样本的label，取label中出现最多的类别作为预测样本的类别。
#  通过矩阵乘法简化计算。
#  
#  优点：对异常值不敏感，无数据输入假定。
#  
#  缺点：计算量大复杂度高。
#  
#  以鸢尾花数据集为例进行算法实现与测试。

# In[1]:

import numpy as np
import pandas as pd
import operator


# In[2]:

train_data = pd.read_csv("F:\Datamining\ML\KNN\data\iris_training.csv")
test_data = pd.read_csv(r"F:\Datamining\ML\KNN\data\iris_test.csv")[10:11]


# In[3]:

# train_data


# In[4]:

# train_data.values[:,4:5].tolist()


# In[5]:
# 机器学习实战原代码，python3个别有调整，输入仅支持一个样本
def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
#         print(int(votelabel))
        classCount[int(votelabel)] = classCount.get(int(votelabel),0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True) #python3 字典iteritems 变为items
    return sortedClassCount[0][0]


# In[6]:

a= classify(test_data.values[:,:4],train_data.values[:,:4],train_data.values[:,4:5],30)
a


# In[7]:
#调整输入支持多个样本
def clac_distance(train_data,test_data,k):
    #label = 
    tile_train_data = np.tile(train_data,(test_data.shape[0],1))
    tile_test_data = np.repeat(test_data.values,train_data.shape[0],axis=0)
    dis = np.sqrt(np.sum(np.square(tile_test_data[:,:train_data.shape[0]-1] - tile_train_data[:,:train_data.shape[0]-1]),axis=1))
    distance = dis.reshape(test_data.shape[0],train_data.shape[0]) #测试样本对应多个训练样本，测试样本距离reshape之后是该测试样本对应的距离
    sink  = np.argsort(distance)
    res = {}
    for i in range(test_data.shape[0]): #处理多维索引
        label_count = train_data['Species'].reset_index(drop=True)[sink[i]][:k].value_counts()
        label = label_count.argmax()
        res[i+1] = label
    return res


# In[8]:

res = clac_distance(train_data,test_data,60)
res




