## Simple Linear Regression ##

## importing libraries ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing  dataset ##
dataset = pd.read_csv("Salary_Data.csv")

## segragating the values ##
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values  

## splitting the dataset into Test / Train set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=1/3, random_state=0)

## applying simple linear regression on dataset

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)

## predicting the output 
y_pred = LR.predict(X_test)

## visualization 

plt.scatter(X_train,y_train,color = 'blue')
plt.plot(X_test,y_pred,color = 'red')
plt.scatter(X_test,y_test,color = 'green')

plt.title("Experience Vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")

plt.show()
