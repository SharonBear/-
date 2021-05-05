# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:20:35 2020

@author: glori
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:29:29 2020

@author: glori
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv('pulsar_stars.csv')
print(data.head)
feature = [' Mean of the integrated profile',' Standard deviation of the integrated profile',' Excess kurtosis of the integrated profile',' Skewness of the integrated profile',' Mean of the DM-SNR curve',' Standard deviation of the DM-SNR curve',' Excess kurtosis of the DM-SNR curve',' Skewness of the DM-SNR curve']
X = pd.DataFrame(data, columns=feature)
target = pd.DataFrame(data,columns=['target_class'])  
y = target['target_class']

XTrain, XTest, yTrain, yTest = train_test_split(X,y, test_size=0.33,random_state=0)


lm = LinearRegression()
lm.fit(X, y)

logistic = LogisticRegression()
logistic.fit(X,y)
print("迴歸係數:", logistic.coef_ )
print("截距:", logistic.intercept_ )

pred_test = logistic.predict(XTest)

Ks=range(1, round(0.002*len(XTrain)+1))
accuracies=[]
for k in Ks:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    accuracy = knn.score(XTest, yTest)
    accuracies.append(accuracy)
    
plt.plot(Ks, accuracies)
plt.show()

pred_test = knn.predict(XTest)
print('預測分數:',lm.score(XTest,pred_test))
print('訓練模型分數:',lm.score(XTest,yTest))
print('資料集的欄位:',data.columns)
