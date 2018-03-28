# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:41:56 2018

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
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
################



df_train=pd.read_csv('D:data_car.csv')
df_train.columns


######################################  1) missing data


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
# Market Category missing_data more than 30%. Analysis correlation between MSRP
# and Market Category using plot


var="Market Category"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.violinplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
# according to the plot, market category doesnt have strong correlation with MSRP
# drop column"Market Category"
del df_train['Market Category']   

# for Engine HP, Engine Cylinders, Number of Doors replace NaN by mean
#mean=np.mean(df_train["Engine HP"][df_train["Engine HP"]!=None])
df_train=df_train.fillna(df_train.mean()["Engine HP":"Engine HP"])

df_train=df_train.fillna(df_train.mean()["Engine Cylinders":"Engine Cylinders"])

df_train=df_train.fillna(df_train.mean()["Number of Doors":"Number of Doors"])

# Engine Fuel Type only has 3 Nan, we can drop these three samples
df_train=df_train.dropna()
###



####################
# dtypes of features
df_train.dtypes

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,cbar=True, annot=True, vmax=.8, square=True)
# according to the correlation plot,
# Engine HP, Engine Cylinders, Year are important features

#######################################     2) select correlated variables
# analysis relationship with categorial variables

var="Engine Fuel Type"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
# choose Engine Fuel Type
 

var="Make"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
# choose Make


var="Model"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
# there are almost 1000 different model type so drop Model


var="Driven_Wheels"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
# no significant difference between differen wheels, drop wheels

var="Vehicle Size"
data=pd.concat([df_train['MSRP'],df_train[var]],axis=1)
fig=sns.boxplot(x=var,y='MSRP',data=data)
fig.axis(ymin=0,ymax=800000)
#no significant difference between differen vehicle size, drop it

# Overall, we choose the engine fuel type, and the engine cylinder,
# engine HP, the make, and the year as the features considered in the Model


#######################################     3) Outliers

#The primary concern here is to establish a threshold that defines an observation as an outlier.
#To do so, we'll standardize the data. 
#In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.
#standardizing data
MSRP_scaled=StandardScaler().fit_transform(df_train['MSRP'][:,np.newaxis])
sns.distplot(MSRP_scaled)  

low_range = MSRP_scaled[MSRP_scaled[:,0].argsort()][:10]
high_range= MSRP_scaled[MSRP_scaled[:,0].argsort()][-55:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
# more than 50 samples whose normalised MSRP more than 5%
# we need to concern


#bivariate analysis MSRP/Engine HP
var = 'Engine HP'
data = pd.concat([df_train['MSRP'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='MSRP', ylim=(0,800000));
# no much outliers

#bivariate analysis MSRP/Engine Cylinders
var = 'Engine Cylinders'
data = pd.concat([df_train['MSRP'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='MSRP', ylim=(0,800000));
# no much outliers


#######################################     4) data normalization 

# test MSRP
sns.distplot(df_train['MSRP'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['MSRP'], plot=plt)
# DO NOT NORMALIZE, IT WILL BE WORTH

# test Engine HP
sns.distplot(df_train['Engine HP'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['Engine HP'], plot=plt)
df_train['Engine HP'] = np.log(df_train['Engine HP'])
# little better...

###############################################################################
#######################Modeling: SVM, RANDOM FOREST############################

#SELECT FEATURES:

df=df_train[["Make","Engine Fuel Type","Engine HP","Year","Engine Cylinders","MSRP"]]

   

#DUMMY VARIABLES:
df = pd.get_dummies(df)   
    
####### split data into train and test sets
x1, x2, y1, y2 = train_test_split(df.iloc[:,:-1], df["MSRP"],test_size=0.2, random_state=0)

# SUPPORT VECTOR REGRESSION 

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

y_SVpred = svr_rbf.fit(x1, y1).predict(x2)

# root mean square error
sqrt(mean_squared_error(y2, y_SVpred)/len(y2))


# RANDOM FOREST REGRESSION

regr = RandomForestRegressor(max_depth=2, random_state=0)

regr.fit(x1, y1)

y_RFpred=regr.predict(x2)  #RF better

sqrt(mean_squared_error(y2, y_RFpred)/len(y2))