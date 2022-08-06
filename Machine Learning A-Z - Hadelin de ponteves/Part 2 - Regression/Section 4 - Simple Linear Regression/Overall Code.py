
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values  # The reason we don't write 0 is because if you see in the variable explorer, you will find out that the size of X is is (30,1) which indicates it's a "matrix" and if you check out the y, its size is (30,) which indicates it's a "vector". 
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0) #1/3 because it's simple model

# in most of Simple Linear we do not need a feature scaling since some libraries will take care of it
"""
# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)
"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Build an object of the class LinearRegression
regressor.fit(X_train, y_train) # fit is a method

# Predicting the Test set result
y_pred = regressor.predict(X_test) # predict is a method

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs. Exprience (Training set)')
plt.xlabel('Years of Exprince')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs. Exprience (Test set)')
plt.xlabel('Years of Exprince')
plt.ylabel('Salary')
plt.show()