
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Training set 
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler    # From two options of Standardisation and Normalisation, the scond one gives a better result. another reason for choosing the second option is because in LSTM we use several sigmoid activation function and since the sigmoid function takes values between 0 and 1 then we would rather to scale our values between 0 and 1 as it is the case in normalisation.
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)  # We doing the feature scaling just for training and at the end we will change it back to its original value.

# Getting the inputs and outputs
X_train = training_set[0:1257]  # The input is not the time but stock price at times t
y_train = training_set[1:1258]  # The outout is the stock price at times t+1

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1)) # Now we have a 2 dimensional array in our X_train. first observation corresponce to the observations, second observation corresponse the features which is stock price at time t. in here we want to add another dimension which corresponse to the timestep. timestep is 1 because our input is stock price at time t and our output is stock price at time t+1 so (t+1)-1=1 # Now our first dimension corrensponse to the observations, second corresponse to the timestep, third corresponce to the features.

# Part 2 - Building the RNN

# Importing the Keras librareis and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential() # Because we don't predicing the categorical data

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))  # Units is the number of memory units. the reason we choosing 4 is because we used in parameter tunning and also it's common practice to use 4 memory units in LSTM # activation function which it can be sigmoid or hyperbolic tanget # the input_shape argument will be None and One; None is just to specify that the mode can expect any timestep, and one because we have one feature. the first element correspose to the timestep and the second element is the number of features we have.

# Adding the output layer 
regressor.add(Dense(units = 1)) # Output of stock price at time t+1

# Compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error') # There is two option for the optimizer: RMS and adam. they have more or less same result and since in here RMS is asking much more memory than the adam optimizer, we choose RMS. but in general first try RMS. RMS is also recomended in the Keras library. # loss: since we have a regression right now then we don't use binary_crossentropy. we use mean squared error. 

# Fitting the RNN to the Training set 
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs) # Feature scaling
inputs = np.reshape(inputs, (20, 1, 1))  # Change the dimension to 3
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # To scale it to its original price

# Visualising the results
plt.plot(real_stock_price, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Homework

# Getting real stock price of 2012 - 2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values

# Getting the predicted stock price of 2012 - 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualising the results
plt.plot(real_stock_price_train, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error    # This is evaluating for the regression model. the model presented in here, is usually the same in other regression problems as well
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))  # in here we got 3.4 which is a number and it doesn't mean so much to us. so we want to convert it to the percentage. in general below 1% is a good model
# the real_stock_price gets a value from 778 up to (about) 830 so all the values are around 800. percentage = rmse/800 which is around 0.4%
