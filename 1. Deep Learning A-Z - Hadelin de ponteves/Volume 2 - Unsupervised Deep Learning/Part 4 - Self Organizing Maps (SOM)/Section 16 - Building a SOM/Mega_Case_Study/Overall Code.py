# Mega case study - make a hybrid (ANN, SOM) deep learning model        # To predict the probability that each customer cheated so we should go from unsupervised to supervised learning

# Part 1 - Identify the frauds with the Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()  # selecting the outline neuron is kind of arbitarary and it actually depends on the threshold.either we want to get the widest which is only number 1 (white color). or we can decrease the theshold a little bit and consider a number a little below 1 as well.

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)  # Me: Remmber everytime you run there is going to be a diffrent nuber for this row.
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from Unsupervised to Supervised deep learning          # For doing so we need a dependent variable since in unsupervised learning, we didn't need it

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # In here we don't need the customer ID and the reason for using the last column is because it might be a relavant information that helps to find some correlation between customer's information and its probability to cheat.

# Creating the dependent variable                                       # 0 means no fraud, 1 means it's fraud
is_fraud = np.zeros(len(dataset))        # first we are going to initialize a vector of zeros (# 690). at the begining we pretend all the customers didn't cheat and then we will extract these customer IDs. for this customer IDs we put a 1 in our vector of zeros for the index corresposing to these customer IDs
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:  # we have to check if the customer's ID is inside list of frauds.
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2) # The reason for choosing the low number is because our dataset is so simple and 1 or 2 epook is totally enough

# Part 3 - Making predictions and evaluating the model

# Predicting the probabilties of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis = 1)  # To add customer ID next to the y_pred # the reason we add 1 at dataset.iloc[:, 0:1].values is because we want a 2 dimension so we can concat it to y_pred since y_pred is 2 dimension
y_pred = y_pred[y_pred[:, 1].argsort()] # Sorting our y_pred # inide bracket y_pred[] we specify what column we want to sort. the reason we don't use just sort alone is because it will sort only one column and not the other one # inside the bracket y_pred[y_pred[]] we specify the index of the column which is 1


