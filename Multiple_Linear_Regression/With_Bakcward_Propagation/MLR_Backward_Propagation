'''## Multiple Linear Regression ##'''

'''## importing libraries ##'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''## importing  dataset ##'''
dataset = pd.read_csv("50_Startups.csv")

'''### segragating the values ##'''
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values  

'''### label encode the state columns ##'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_x = LabelEncoder()   
X[:,3] = label_x.fit_transform(X[:,3])
hotenc = OneHotEncoder(categorical_features=[3]) ## categorical feature == Column number you want to onehotencode
X = hotenc.fit_transform(X).toarray() ## convert to array else cant be seen in the variables

'''## Removing extra dummy variable ##'''
X = X[:,1:]

'''### splitting the dataset into Test / Train set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=0)

'''### Fitting the Multiple linear Regression model on the Data set ##'''

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train,y_train)

'''## Predicting the Output ##'''

y_pred = LR.predict(X_test)


'''## Backward Eliminination to find out which variable effects the 
   model maximum ##'''

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
LR_BE = sm.OLS(endog=Y,exog=X_opt).fit()
print(LR_BE.summary())
