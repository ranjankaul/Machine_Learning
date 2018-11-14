## Data pre-processing template ##


## importing libraries ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## importing  dataset ##
dataset = pd.read_csv("Data.csv")

## segragating the values ##
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values  

## editing the missing data ##
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
X[:,1:3] = imputer.fit_transform(X[:, 1:3])

## One hot encoding the categorical data ##
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_x = LabelEncoder()   
X[:,0] = label_x.fit_transform(X[:,0])

hotenc = OneHotEncoder(categorical_features=[0])
X = hotenc.fit_transform(X).toarray() ## convert to array else cant be seen in the variables

label_y = LabelEncoder()
Y = label_y.fit_transform(Y)

## splitting the dataset into Test / Train set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

## Feature scaling 

from sklearn.preprocessing import StandardScaler
sc_X =  StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

