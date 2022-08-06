
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding the categorical data     # we have two columns of categorical which we want to turn it to the numerical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1]) # Creating dummy variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] # Avoiding dummy variable trap

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now let's make the ANN!

# importing the Keras libraries and packages
import keras
from keras.models import Sequential # To initialize our neural network
from keras.layers import Dense # Creating the layers in our artificial neural network  # Step 1 of algorithm is done thanks to the dense function. Step 2, we have 11 inout notes. For step 3 best activation function for the hidden layer is the rectifier function. the sigmoid function is reall good for the output layer since it will get the probabilities of diffrent segments          

# Initialsing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer 
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11)) # This adds hidden layer # units is the number of notes in the hidden layer. there is no rule for choosing the best number but based on exprience: the average of the number of nodes in the input layer and the number of nodes in the output layer. if you don't want to use this tip and you want to choose this number (choosing the number is like the art) then you should expriment with a technique called parameter tuning which is about using some techniques like k fold cross validation. """units: number of nodes in hidden layer which is input nodes + output nodes / 2. 11+1/2""" """kernel_initializer: Step 1 which initializes the weight randomely and make sure that weights are given small numbers close to zero""" """activation: Stands for rectifier """                               

# Adding the second hidden layer       """ this step wasn't so necessary for this dataset but we doing it anyway"""
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # in here, if we dealed with three or more than three category then we change the units and also change the activation to the soft max. soft max is singmoid which is applied to more than two category.

# Compile the ANN                      """In here we are applying stochastic gradient descent on the whole ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  #""" optimizer: We have initialized the weights but it also requires a algorithm which is stochastic gradient descent and a very efficient one is called Adam """ """loss: corresponce to the loss function within the stochastic gradient descent algorithm that is within adam algorithm. this is more mathemtical matter (logarithmic loss). if the dependet variable was binary then this logrithmic loss function is called binary_crossentropy. and if our dependent variable has more than two outcome (2 not included) then we use categorical_crossentropy""" """metrics: ceriterion to evaluate the model. in here we use accuracy which what happens is that after each observation or each bach of many observation when the weights are dated. the algorithm uses this acuracy criterion to imporove the models performance. remmber that metrics expecting a list"""

# Fitting the ANN to the training set        """ Step 6 and Step 7""" """epoch: the number of times we plan to train the ANN in the whole training set """ """batch size is the number of observations after you wanna update the wights"""
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100) # if you check out, our accuracy rate is around 86%

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set result
y_pred = classifier.predict(X_test)  
y_pred = (y_pred > 0.5)    # In here we want to predict 1 for over the threshold and 0 below the threshold

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   # Accuracy = (1555+133) /2000

