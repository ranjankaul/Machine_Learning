## Decision Trees ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


## import the dataset ##
dataset =  pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

## Predicting Result ##

y_pred = regressor.predict(6.5)

## high resolution curve plotting ##

X_grid = np.arange(min(X),max(X),0.01) 
X_grid = X_grid.reshape((len(X_grid),1))    
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('estimation of salary vs position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

print(y_pred)  # 150000
