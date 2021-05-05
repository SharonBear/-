# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:22:09 2020

@author: glori
"""

import pandas as pd
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv('insurance.csv')


data.sex.replace(['female','male'],[1,2], inplace = True)
data.smoker.replace(['no','yes'],[1,2], inplace = True)
data.region.replace(['southwest','southeast','northwest','northeast'],[1,2,3,4],inplace = True)
print(data.head)
feature=['age','sex','bmi','children','smoker','region']
X = pd.DataFrame(data, columns=feature)
target = pd.DataFrame(data,columns=['charges'])  
y = target['charges']

XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.33,random_state=0)

lm = LinearRegression()
lm.fit(X, y)
print('迴歸係數 : ', lm.coef_)
print('截距 : ', lm.intercept_)


Ks=range(1, round(0.2*len(XTrain)+1))
accuracies=[]
for k in Ks:
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    knn.fit(X,y)
    accuracy = knn.score(XTest, yTest)
    accuracies.append(accuracy)
    
plt.plot(Ks, accuracies)
plt.show()

pred_test = knn.predict(XTest)

print('預測分數:',lm.score(XTest,pred_test))
print('訓練模型分數:',lm.score(XTest,yTest))
print('資料集的欄位:',data.columns)