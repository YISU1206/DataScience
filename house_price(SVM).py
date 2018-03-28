# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:48:46 2018

@author: 喵大人
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler  #处理异常值， 将异常值标准化工具
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
################
df_train=pd.read_csv('D:train_house.csv')
df_train.columns


# analysis SalePrice
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
# deviate from normal distribution 不遵循正态分布
# have appreciable postive skewness 有明显偏移
# show peakedness 有峰值

##############################################################确定几个自变量
# analysis relationship with numerical variables

var='GrLivArea'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#linear relationship

var='OverallQual'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
#linear relationship

var='YearBuilt'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)

#神兵利器：heatmap 显示所有变量之间的correlation
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice的heatmap
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws=
                 {'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#通过上面的分析我们可以从中找到有几个自变量has strong correlation to others, so we just
#need to pick one of them.

#########最后我们选定的变量scatter图：
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 
       'TotalBsmtSF',  'FullBath',  'YearBuilt']
sns.pairplot(df_train[cols],size=2.5)
plt.show()




###################################################################开始处理自变量缺失问题
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)
#原则上是删除缺失数据超过15%的自变量。其他的缺失变量发现都与其他的不缺失变量有关所以删除他们不会对
#最终结果产生影响， 除了‘Electrical’只有一个缺失数据就把那一条数据删除吧

df_train=df_train.drop((missing_data[missing_data['Total']>1]).index,axis=1)
df_train=df_train.drop(df_train[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()     #检查是否没有missing data

###################################################################异常值

#1. 第一种， 但从数上面考虑： 发现outlier方式： 正则化


#数据异常值的清理就是设定一个阈值（threshold)， 具体将数据标准化处理（data standardize）也就是将数据
#变成mean=0,std=1的数， 两段越过5%的数算作异常值
#standardizing data
saleprice_scaled=StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
sns.distplot(saleprice_scaled)  #standardize后的结果

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#负数还好， 但是有最大超过7的数， 我们先放这里再考虑， 不一定真的违背常理

###########

#2. 第二种： 从图上考虑， value和saleprice
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#从图中显示， 有两个房子面积很大但是价格便宜的离谱， 不能归为普遍情况； 另外最大值上面价格也很贵， 但是
#走向是和其他的一样的， 所以keep
#deleting points
#正好这两个特殊值是GrLivArea最大的两个值， 如果在大部队trendline 的下面不太好删除
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#

#########################################################对变量normal的研究
#如果变量是normal， 可以避免很多问题：异方差问题heteroscedasticity等等。。。

#1.研究saleprice异方差性

#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#我们发现， saleprice遵循正偏态positive skewness，在这种情况下用 log transformations 可以解决
#applying log transformation
#log前提就是一定没有0或者负值
df_train['SalePrice'] = np.log(df_train['SalePrice'])

############

#2.研究GrLivArea异方差性

#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation 同样~~~
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])


###########

#3.研究TotalBsmtSF异方差性

#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#有问题了， 因为有0， 不能用log， 解决方式就是将非零的log，0的就放在那儿吧。。。

df_train['HasBsmt']=pd.Series(len(df_train['TotalBsmtSF']),index=df_train.index)
df_train['HasBsmt']=0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt']=1

#transform data
df_train['TotalBsmtSF'][df_train['HasBsmt']==1]=np.log(df_train['TotalBsmtSF'])


#变身后的图：我们只看>0的同志
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


############################################################同方差性 homoscedasticity
#同方差性即在linear regression 中noice本身应该cov(noices)=0
#如果解决之前的变量normal的问题， 很大程度上可以解决同方差性问题
#最好展示homoscedasticity的方式就是图表
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
#这两个都没有问题



############################################################将类别变量转化，类似onehot


df_train = pd.get_dummies(df_train)















