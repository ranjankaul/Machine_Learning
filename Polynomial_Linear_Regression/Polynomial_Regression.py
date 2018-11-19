'''### Polynomial Regression ###'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Position_Salaries.csv')
 
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

'''## As the dataset isnt big hence not using test-train split ##''' 
'''## Feature scaling already done by the linear regression module ##'''

'''## fitting linear regression model ##'''
from sklearn.linear_model import LinearRegression
Lin_LR = LinearRegression()
Lin_LR.fit(X,Y)

'''## Fitting the polynomial Regression model ##'''
from sklearn.preprocessing import PolynomialFeatures
Poly_LR = PolynomialFeatures(degree=4) # Degree makes a polynomial matrix of the input values 
X_poly = Poly_LR.fit_transform(X) 

Lin_LR2 = LinearRegression()
Lin_LR2.fit(X_poly,Y)                  # Adding the polynomials to linear Regression 


'''## Visualising the Linear Regression Graph for dataset ## '''
plt.scatter(X,Y ,color ='blue')
plt.plot(X,Lin_LR.predict(X),color ='green')
plt.title("Linear Regression Salary Estimate")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()

'''## Visualising the polynomial Linear Regression Graph for dataset ## '''

plt.scatter(X,Y ,color ='blue')
plt.plot(X,Lin_LR2.predict(X_poly),color ='red')
plt.title("Polynomial Regression Salary Estimate")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()


print(Lin_LR.predict(6.5))
print(Lin_LR2.predict(Poly_LR.fit_transform(6.5)))
