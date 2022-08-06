# In here we are implementing a model for fraud detection. these are the information customer should provide when filling the application. in here we don't build a supervised model to predict whether each customer cheated or not. in here we doing a unsupervised learning which it means we identify some patterns and one of these patern going to be a potential fraud. when we think about fraud, we think about outliers since most of credit card clients follow the rules. to detect the outliers we need MIT (something like mean in turnour neuron distance). it means that in our SOM for each neron, compute the mean of Euclidean distnace between this neuron and its neuron to its neighberhood. we define the neighberhood manually. and we define a neighberhood for each neuron and we do the mean of Euclidean distnace. by doing this we can detect the outliers because outliers will ber far from allneuron in its neighberhood.


# Importing the librareis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv') # For privacy the names and values has been changed to a meaningless symbols.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # Remmeber we only use this on supervised learning and not in here. we just doing this to make a distiction between the approved and not approved customer


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # We want the range between 0 and 1
X = sc.fit_transform(X)


# Training the SOM
from minisom import MiniSom  # This is a library developed by another developer. the file of it, is in the same directory.
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) # our first arguments are x and y which is dimension of grid. the number is arbitarary. to be very accurate with search of outliers make a bigger map but in here we don't have that much observation (customers) so we make a 10 by 10 grid # input_len would be 14 (attributes) + 1 (customer ID) = 15. one in here is customer id which we actually don't need it but we use it so we can identify the cheaters # sigma is the radius of diffrent neighberhood which is defult by 1 # learning_rate: bigger, faster will be build. smaller, slower it will be build which we leave it to default of 0.5 # decay_function is used to improve the convergence but here we leave it to None and not using it #
som.random_weights_init(X)  # Inisialising the weights based on the 'Training the SOM' picture# the method build by developer
som.train_random(data = X, num_iteration = 100) # Method to trainin the SOM # num_iteration: number of iteration we want to repeat the step 4 to 9.
 

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show    # MID mean into neuron distance: mean of the distances of all neurons around the winning node inside the neighborhood we defined. # High MID = the more, winning node far away from its neighbors inside a neighborhood in other word the higher MID the more the winning node is an outlier. we don't do this with figures, we do it with colors so the more the color is white, the higer MID will be.
bone() # In her we initialising the figure (window that contains the map)
pcolor(som.distance_map().T)  # To put diffrent winning node on the map. # in fact map method will return all the mean into neuron distance in one matrix # T is because we want to get the transpose of MID matrix
colorbar() # to get the legend of the color. for instance being white or dark is represantative of what # On right han side of plot we see range values of MID but these are normalized values from 0 to 1 # Highest MID in this plot corresponce to the white color so the white colors are outliers 
markers = ['o', 's'] # in here we add some markers to tell whether each of these modes got approval or not # We create 2 markers: 1- red circle: customers who didn't get approval 2- green squares: customers who got approval 
colors = ['r', 'g']

for i, x in enumerate(X):  # i: diffrent values of diffrent indexes of our customer which i takes value from 0 to 689 # x: diffrent vectors of customers so x start to be equal to first vector (which is first row) and at the next iteration it's going to be second row or customer and so on  # inside enumerate we add x which contains all our customers
    w = som.winner(x) # w is the winning node and we actually have a method for that
    plot(w[0] + 0.5,  # first two are x and y coordinate of winning node. the reason we add 0.5 is because without it, it's the coordinate of lower left corder of square and we want to put it in the center # Place marker for this winning node 
         w[1] + 0.5,
         markers[y[i]],  # In here we want to know wether the marker going to be red circle or green squared. we do it with the help of y that we define at the start. # i is the index of customer, y[i] is the dependent variable for that customer that is 0 if not approved  and 1 if got approval. for example if y[0] then marker[0] is circle.
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)  # In here, not only we see diffrent MIT for all winning node, and we also see the customer associated to the winning nodes are customers who got approval or not
show()  # red circle: customer associated with this winning node didn't get approval # green square: customer associated with this winning node got approval # red and green together (in white area only): they are outliers and have high chance of fraud


# Finding the fraud                                                     # For doing so we don't have a inverse maping function to get the list (in general it's possible but not in here). so first we get all these mapings in dictionary from the winning nodes to the customer and then using the coordinate of outliers that we identified (white ones) 
mappings = som.win_map(X) # Key or first column is our coordinate. coordinate (0,0) indicades the lower left corner. # Size: number of customers associated wo this winning node. # double-click on values: gives all of those customers. each row is one customer. # double-click again on values: it corresponse to the attribute of this customer with the first attribute which is customer id. (the number is scaled. at the end we use inverse scaled method to get the real value)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)  # in here we get the coordinates of two outliers (white one) and concate it in here. for doing so visualize the SOM and get the coordinate (row and column) of those two nodes. # this is a list of potential cheaters
frauds = sc.inverse_transform(frauds)  # to reverse the scaling


