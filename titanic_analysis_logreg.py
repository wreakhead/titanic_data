# -*- coding: utf-8 -*-
"""
Created on Fri May 31 02:32:19 2019

@author: IMRAN
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression

titanic_data=pd.read_csv("titanic.csv")
titanic_data.head(10)
print("np of passengers "+ str(len(titanic_data.index)))

#data anaylysis
sns.countplot(x='Survived',data=titanic_data)
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
titanic_data['Age'].plot.hist() 
titanic_data["Fare"].plot.hist(bins=20,figsize=(10,5))
print(titanic_data.info())
sns.countplot(x="SibSp",data=titanic_data)
sns.countplot(x="Parch",data=titanic_data)

#data wrangling
titanic_data.isnull()
titanic_data.isnull().sum()
titanic_data.head()
print(titanic_data.drop('Cabin' , axis=1,inplace=True))
#too much missing values
titanic_data.dropna(inplace=True)
titanic_data.isnull().sum()

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)

pd.get_dummies(titanic_data['Age'])
#testing
embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
#if we do not drop 1st i.e 'C' will be present
#dropping 'C' means when 'Q=0 & S=0' that will be 'C'
pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
titanic_data=pd.concat([titanic_data,sex,embark,pcl],axis=1)
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)

print(titanic_data.info())

#train data
X=titanic_data.drop('Survived',axis=1)
#indepedent
y=titanic_data['Survived']
#dependent variable
from sklearn.cross_validation import train_test_split

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

logmodel=LogisticRegression()

print(logmodel.fit(X_train,y_train))

predictions=logmodel.predict(x_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))




