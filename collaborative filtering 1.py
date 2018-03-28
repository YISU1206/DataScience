# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:24:20 2017

@author: 喵大人
"""

import numpy as np
import pandas as pd
header=['user_id','item_id','rating','timestamp']
df=pd.read_csv('D:u.data',sep="\t",names=header)
###data analysis   
#user-item matrix
data=df.pivot(index='user_id',columns='item_id',values='rating')    

######################################### Plan B
from sklearn.cluster import KMeans

data=data.fillna(0)
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(data)
# Getting the cluster labels
labels = kmeans.predict(data)

#########################################




# we decide to use pearson to calculate user similarity, 
#To choose min_periods, we need to find one user pair who are most similar

foo = pd.DataFrame(np.empty((len(data.index),len(data.index)),dtype=int),index=data.index,columns=data.index)
for i in foo.index:
    for j in foo.columns:
        foo.ix[i,j] = data.ix[i][data.ix[j].notnull()].dropna().count()
##foo: value=numer of movies which they all marked
for i in foo.index: 
    foo.ix[i,i]=0
ser = pd.Series(np.zeros(len(foo.index)))   
for i in foo.index:
    ser[i]=foo[i].max()
ser.idxmax() #846
ser[846]  #346
foo[foo==346][846].dropna()  #405,346

data.ix[346].corr(data.ix[405])
test = data.reindex([405,346],columns=data.ix[405][data.ix[346].notnull()].dropna().index)

######start to find min_period:
periods_test = pd.DataFrame(np.zeros((40,5)),columns=[10,20,40,60,80])
for i in periods_test.index:
    for j in periods_test.columns:
        sample = test.reindex(columns=np.random.permutation(test.columns)[:j])
        periods_test.ix[i,j] = sample.iloc[0].corr(sample.iloc[1])

periods_test[:5]
periods_test.describe()   # we choose min_period=60

######start to recommend######################
corr = data.T.corr(min_periods=60)  #少于60个和别人有相同评论电影的用户自动过滤
for i in corr:
    corr.ix[i,i]=None
corr_clean = corr.dropna(how='all')
corr_clean = corr_clean.dropna(axis=1,how='all')
## CHOOSE ONE USER RANDOMLY
lucky = np.random.permutation(corr_clean.index)[0]
#在这里随机选择的选手是60人以上有重合marked movie
gift = data.ix[lucky]
gift = gift[gift.isnull()]#现在 gift 是一个全空的序列
###填充这个 gift:预估具体某一位user对所有电影的打分
corr_lucky = corr_clean[lucky].drop(lucky)#lucky 与其他用户的相关系数 Series，不包含 lucky 自身
corr_lucky = corr_lucky[corr_lucky>0.3].dropna()#筛选相关系数大于 0.1/0.3 的用户
#在之前去掉很多选手自身以及无交集选手以及相似度<0.1

#######塑造分数：
for movie in gift.index:#遍历所有 lucky 没看过的电影
    prediction = []
    for other in corr_lucky.index:#遍历所有与 lucky 相关系数大于 0.1/0.3 的用户
        if not np.isnan(data.ix[other,movie]):
            prediction.append((data.ix[other,movie],corr_clean[lucky][other]))
    if prediction:
        gift[movie] = sum([value*weight for value,weight in prediction])/sum([pair[1] for pair in prediction])
  
gift.dropna().sort_values(ascending=False)[:10]



