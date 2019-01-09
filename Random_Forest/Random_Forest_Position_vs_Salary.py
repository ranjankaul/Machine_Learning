## Random forest ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
    
dataset =  pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

## Random Forest Regressor ##

regressor = RandomForestRegressor(n_estimators = 1000 , random_state=0)
regressor.fit(X,Y)

print(regressor.predict(6.5))

## Predict ##

X_grid = np.arange(min(X),max(X),0.01) 
X_grid = X_grid.reshape((len(X_grid),1))    
plt.scatter(X,Y,color = 'green')
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('estimation of salary vs position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
