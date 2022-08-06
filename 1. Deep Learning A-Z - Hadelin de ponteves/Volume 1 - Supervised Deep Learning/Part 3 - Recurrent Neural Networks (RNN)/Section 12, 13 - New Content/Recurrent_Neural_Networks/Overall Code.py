
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # .values makes a numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output                      # Getting the 60 previous steps including the current step (time t) to predict the next step (only one step) at time t+1
X_train = [] # Will include the previous 60 stocks prices
y_train = [] # Will include the stock price the next financial day
for i in range(60, 1258): # The reason for starting at 60 is because for predicting at start we need to have the previous 60 stock prices. 
    X_train.append(training_set_scaled[i-60:i, 0]) # So in here since we start at 60 then we are going to start predicting from 61 to the end # 0 is the the index of column which is the 0
    y_train.append(training_set_scaled[i, 0]) # this is the prediction at time t+1 but remmber that this is i and not i+1 becuse the indexes in out dataset start at 0 and so 61 have the index 60 in there.
X_train, y_train = np.array(X_train), np.array(y_train) # So far out X_train and y_train were list and we want numpy array # if chekout the X_train every row correspoce to the 60 stock prices and after the first row, the last column in each row is equal to prediction of algorithm which is also in y_train

 
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Adding more dimension (with rehsape method) that is the number of predictors we can use to predict what we want which in here predictors are indicators. so far we had the 'open' indicator. by adding more indicator we can predict better the upward or downward trent of google stock price # In here we are converting 2 dimension array to 3 dimension # for arguments of 3D tensor in RNN check out keras.io in part RNN. arguments: 1- batch_size which is the total number of stock prices from 2012 to 2016. in here we have 1198 which is the number of rows in X_train  2- timesteps which is number of timesteps. in here it's 60 or number of columns in X_train  3- input_dim which are the predictors or indicators in here like 'open' or 'close'. in here it's 1 because it's only one indicator but change this if it's more thatn 1. in another example we can say that if apple wouple buy some of their chips from samsung then these two stock prices are highy correlated and both of them depend on each other (one for the chips and the other one for the money) which we can use here as well.                                                                                                                   


# Part 2 - Building the RNN

# Importing the Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout  # Using regularization to avoid overfitting

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # arguments: 1- units: number of LSTM itself or the memory units 2- return_sequence that we put True because we are building a stacked LSTM which have serveral LSTM layers. at the end when we are not going to add another layer we put this to False or we don't write anything since it's the default 3- input_shape that is the shape of the input containing X_train that we have created in the last step of the data preprocessing part. it's a input shape in 3D containing: observations, timesteps, and the indicators. since observation in here automaticly taken into account. so we only put the last two in here. but remmber that we still have this 3D structure in here.
regressor.add(Dropout(0.2)) # argument: 1- dropout rate that is the rate of nerons you want to drop that is you want to ignore. a classic number is 20% of the neron in the layer. # 20% x 50 = 10 neuron will be ignore and dropout


# Adding the second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2)) 

# Adding the third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2)) 

# Adding the forth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50))  # return_sequences in here is going to be false
regressor.add(Dropout(0.2)) 

# Adding the output layer
regressor.add(Dense(units=1))

# Comiling the RNN
regressor.compile(optimizer = 'adam', loss='mean_squared_error')  # for optimizer in RNN RMSprop is recomended. by expriencing adam seems like a better choise # The reason for chooing MSE is becuase we have regression in here and it's not classification anymore

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) # Later put this on the loop and also calculate the error on test set as well for finding the best parameters


# Part 3 - Making the predictions and Visulizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values # We could also call this test_set

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # Firstly we need the first 60 previous financial day before the actual day for predicting the january 2017. # In order to get each day of January 2017, the 60 previous stocks prices of 60 previous days, we need the both training set and test set (december data is in training set and in test set we need also january). so we will do the concatenation. in here we don't concatenate the training_set and real_stock_price (test set) because it leads a problem that then we have to scale this concatenation of training set and test set. with doing so we would change the actual test values. so in here we make a new concatenation. the reason why we do scaling in here is becuase we have trained our model in scaled values. # axis 0 is vertical
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # getting the 60 previous stock price which in financial days in euql to 3 months (weekends doen's count) # first we substrack the dataset_total from dataset_test in order to get the first day of january which in here is 3rd day. after ward substracting it from 60  # writing .values to make this a numpy array
inputs = inputs.reshape(-1,1) # sice in previous step we didn't use iloc, then we didn't get the numpy array shape as well.  # gettng the values in lines and in one colomn
inputs = sc.transform(inputs)
X_test = [] # we don't need y_test because we are not doing any training and we are doing the predictions directly # You can copy pase this from above and make some changes to it
for i in range(60, 80): # The reason we are doing 80 instead of 1258 is becuase the test set contains 20 financial days so 60 + 20 = 80
    X_test.append(inputs[i-60:i, 0]) 
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # The 3D structure # Compy pasting from above
predicted_stock_price = regressor.predict(X_test) # Predicting the X_test 
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Since so far we have worked with scaled data then at the end we want to scale back the data to its orifinal value


# Visualizing the results
plt.plot(real_stock_price, color='red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Evaluation
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

"""
Improvement strategies

1 - Getting more training data: 
    we trained our model on the past 5 years of the Google Stock Price but it would be even better to 
    train it on the past 10 years.
    
2 - Increasing the number of timesteps: 
    the model remembered the stock prices from the 60 previous financial days to predict the stock price
    of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to 
    increase the number of timesteps, by choosing for example 120 timesteps (6 months).
    
3 - Adding some other indicators:
    if you have the financial instinct that the stock price of some other companies might be correlated 
    to the one of Google, you could add this other stock price as a new indicator in the training data.
    
4 - Adding more LSTM layers: 
    we built a RNN with four LSTM layers but you could try with even more. 
    
5 - Adding more neurones in the LSTM layers: 
    we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better 
    to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. 
    You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.
"""

# Tuning the RNN
"""
you can do some Parameter Tuning on the RNN model we implemented.

Remember, this time we are dealing with a Regression problem because we predict a continuous outcome (the Google Stock Price).

Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters.

"""