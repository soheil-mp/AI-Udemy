
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting the dataset into Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)"""

# Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results
# This plt does not meet the description of decistion trees because in decision trees for 
# every 1 unit interval, the result equals to averedge of dependent variables which is constant (horizantal line)
"""plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()"""

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
# The above problem can be solved by using a high resolution plot. note that, this is a non-continous regression model
X_grid = np.arange(min(X), max(X), 0.01) # the reason we use 0.01 instead of 0.1 is because in 0.1 the vertical lines are not stricly vertical
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# This code is in 1D, the tutotial is in 2D and best usage is in 3D
