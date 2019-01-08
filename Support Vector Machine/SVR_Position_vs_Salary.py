## Support Vector Regression ##

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# import dataset

dataset =  pd.read_csv("Position_Salaries.csv") 
X = dataset.iloc[: , 1:2].values
Y = dataset.iloc[: , 2:3 ].values

## Feature scaling 
sc_X =  StandardScaler()
X = sc_X.fit_transform(X)

sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

## SVR regressor ##
regressor  = SVR(kernel = 'rbf')
regressor.fit(X,Y)

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(6.5))) ## scaling the value and getting the 
                                                                        ## prediction  and  getting the final value from the inverse.


## plot the SVR Salary vs  Position                                     
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Salary Value")
plt.xlabel("pos")
plt.ylabel("Salary")
plt.show()

print(Y_pred)




