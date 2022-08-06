

# Part 1 - Data Preprocessing


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  # Encoding country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Encoding gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1]) # Creating dummy variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoiding dummy variable trap

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Part 2 - Making ANN


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
classifier.add(Dropout(P = 0.1)) # P: a float between 0 and 1, and it's the fraction of the input we want to drop. for example if we put 0.1 (or 10 percent) then it means that at each iteration one neuron will be disabled. start examining by 0.1 if it did't work well then add another 0.1 at each time so 0.1, 0.2, 0.3, 0.4, 0.5 if the number reaches to the 1 then underfitting will accure so don't go over 0.5

# Adding the second hidden layer with dropout
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(P = 0.1))

# Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)



# Part 3 - Making the prediction and evaluating the model


# Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) # The information should be made in horizontal  # In here we should use a two double brackets. if we only use one double brackets then it's in column. and two double brackets creates two dimentional array. # First thing to do is to check the order of our observation to be same as our actual data # The first two numbers indicate to country (based on the oreder) and since the country is categorical then we write it's dummy variables which is 2 numbers in here (you have to check the dataset with X to figure it out). # The reason we use 'sc' standard scalar is because we only fited our training set and not the test set or other observations. # The reason for putting our first number as a float is because we will getting a warning otherwise. there is not need to convert all numbers to float, only one of them is enough.                 
new_prediction = (new_prediction > 0.5) # False: The custimer will not leave the bank.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # Accuracy = (1548+137)/2000



# Part 4 - Evaluating, Improving and Tuning the ANN


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier # this combines Keras and Scikit learn. in other word we can include K-Fold cross validation in our Keras classifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()  # 83.4%
variance = accuracies.std()  # 0.9%
# We are in Low Bias Low Variance right now


# Improving the ANN
# Dropout Regularization to reduce overfitting if needed                # dropout works in this way: each iteration of the training, some nerons are randomely disabled to prevent them from being dependent on each other when they learn the correlation. so the ANN learns several independent correlation in the data

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier # this combines Keras and Scikit learn. in other word we can include K-Fold cross validation in our Keras classifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=11))
    classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],   # 25 and 32 are choosed based on previous exprience  # In here we want to improve our hyperparameters
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}   # If we want to tune any hyperparameters that are in the architecture of our ANN, we have to create a new argument for the hyper parameter that we want to tune. and then replace the value in our ANN architect by the name of our argument

gridsearch = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)

gridsearch = gridsearch.fit(X_train, y_train)  

best_parameters = gridsearch.best_params_

best_accuracy = gridsearch.best_score_
# imagine there is three medals for imporving: 1. Gold medal with 86% accuracy 2. Silver gold with 85% accuracy 3. bronze medal for 84% accuracy # 86.26% is the accuracy of last round. real measure of accuracy is in best_accuracy which is 85% so in here we got a silver medal. homework is about getting the gold medal.
        
        