
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting the dataset into Training set and Test set
# In here we don't split the data because of our data. if you notice we want the accurate fit for our data and the data is small as well.
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""


# Feature Scaling
# in simple/Multiple/Polynomial Linear Regression, we don't need Feature Scaling because the library do it for us.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)   # start with (degree=2) as default and check the diffrent degrees
X_poly = poly_reg.fit_transform(X) # In here it automaticly added a column of 1's at the start
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression result
plt.scatter(X, y, color="red") # real value
plt.plot(X, lin_reg.predict(X), color="blue") # Predicted values
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression result
plt.scatter(X, y, color="red") # real value
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue") # Predicted values
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

"""
# Visualising the Polynomial Regression result
X_grid = np.arange(min(X), max(X), 0.1) # To have a high-resolution and smoother curve
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color="red") # real value
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color="blue") # Predicted values
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()"""

# Predicting a new resualt with Linear Regresison
#lin_reg.predict(X) #Predicting all 10 levels
lin_reg.predict(6.5) # Predicting Level 6.5

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
