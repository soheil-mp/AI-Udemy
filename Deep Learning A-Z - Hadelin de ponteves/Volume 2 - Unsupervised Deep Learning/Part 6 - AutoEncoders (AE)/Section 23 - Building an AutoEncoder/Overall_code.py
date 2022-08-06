
# Imporint the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # This is a modulo of torch to implement neural networks
import torch.nn.parallel # This is for paralle computation 
import torch.optim as optim # This is for optimizer
import torch.utils.data # This is the tools we use
from torch.autograd import Variable # This is for stochastic gradient descent

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') # sep is our seperator # header: we don't have a header here so we put None # engine: to make sure that dataset is imported correctly # encoding: we put this because of some special charecters in the movies that cannot treatedd properly with the classic encoding UTF 8  # first column is movie id
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') # first column is the user ID # second column is the gender # third column is the age # forth column is some codes that corresponce to the user's job # last column is a code
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') # first column corresponce to the users so the number 1 corresponce to the first user (all of the 1's are same user).  # sceond column corresponce to the movies (movie's ID)  # Third column corresponce to the ratings  # forth column is the time stamps that we don't absoulutly care. this correspoce when the user rated the movie.

# Preparing the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') # in the ml-100k you can see 5 training and test splits of the whole dataset composed 100'000 ratings. u1.base(training_set), u1.test (test_set) until u5.base, u5.test so each of these are seprate trining set and test set. what's the use of these? that's to perform k-fold validation manually. but we are not going to perform k-fold validation in this section so we only perform on one of them # we will convert this to the array because in here we get a data frame and we prefer arrays # seperator in here is a tab, the default is comma. in here instead of 'sep' we use 'delimiter' # In here, training_set is 80% of the original dataset composed of 100'000 so that would be 80 percent 20 percent split which is a optimal train split
training_set = np.array(training_set, dtype = 'int') # Converting data frame to array  # Second argument is the type of this new array, and since we only have user IDs, movie IDs and ratings that are integers then we convert it to an array of integers
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')  # Note that ratings or movies are diffrent in training_set and test_set
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies                                    # The reason we doing so is because in the next part we want to convert the training set and test set into a matrix that one matrix for training set and one matrix for test set which we set its line going to Users, and columns are going to be the Movies and cells are going to be the Ratings. # in each of these matrices we want to include Users and Movies. if the user didn't rate we input 0 in there. each cell of indexes u,i which u indicates number users and i is the movie. each cell u,i get a rating of movie i by user u
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) # The reason we are getting the max from these 2 dataset is because it is possible that each of those are containing the highest user ID. in here it's training set but for other datsets it's possible that, it test_set. # Another reason why we use 'max' is because there are some repetition in the user IDs and with this method we will get the highest number which is equal to total number of users. # the reason why, we used 'int' is because we want to force that we have integers, otherwise we will get an error.
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))  # First column was for users and second column for movies

# Converting the data into an array with users in lines and movies in columns    # The reason we want to convert this is because it is required for the input of restricted boltzmann machine. in general, we create an array with Observations (users) in lines, and Features (movies) in columns.
def convert(data):  # The reason for crating a function is because we want to apply this on training set and test set
    new_data = []  # For creating the array with users in lines, and movies in columns.in here we won't make a 2-dimension numpy array. we will create a list of list. list of list means that we are going to have several lists, one list for each line. since we have 943 users then we are going to have a 943 lists will be a 1682 elements which is the number of movies. # This list of list is our final output. in here we initialize it as a list
    for id_users in range(1, nb_users + 1): # + 1 is because in range upper bound is excluded    
        id_movies = data[:, 1][data[:, 0] == id_users]  # In here we put the movie ID of the specific user that we are in the loop on  # data is training set and test set # Syntax in here: we want the second column of data (training_set or test_set) such that (second bracket) first column (user ID) is equal to 1. # second bracket is not list, it's condition
        id_ratings = data[:, 2][data[:, 0] == id_users] # Rating ID is in third column # this gives us the rating starting from the first user.
        ratings = np.zeros(nb_movies) # Creating a list of rating of the users. be careful that this is not id_rating. because the one that we want to get is not only the rating of the movies the user rated but also zeros when the user didn't rate the movie. so now we want to create the list of 1682 elements and for each of the movies, we get the rating of movie, if it has been rated and 0 if not. so id_rating is the rating of the movie that user rated  # in here we make the zero list and then replace it with the rating 
        ratings[id_movies - 1] = id_ratings  # Replacing the ratings by zeros list. # The trick in here is that the index in python starts at 0 and index at movie ID starts at 1
        new_data.append(list(ratings)) # In her we want to make sure that ratings is list. because we really need a list of list 
    return new_data
training_set = convert(training_set)  # If you check out the old dataset of training_set then you will notice that in the second column for the first user the movie ID starts at 2 so for the new training_set we see 0 for the first one 
test_set = convert(test_set)

# Converting the data into Torch tensors                                    # Soon we will start building the architect of neural network, and we will build this architecture by PyTorch Tensors # Tensors: arrays that contain elements of a single data type. so tensors are a multi-deminstion matrix but instead of being in numpy array, it's in pytorch array. (numpy can be used but it's not so efficient).# for autoencoder, we could even use tenserflow but pytorch gets a better result  # In here traing set going to be one torch tensor and test set going to be another torch tensor. there is going to two seperate multi-dimension matrices based on pytorch
training_set = torch.FloatTensor(training_set)  # FloadTensor: it will create an object of this class. this object will be torch tensor itself. torch tensot is a multi-deminsion matrix with a single type. in here single type is going to be float  # This is not a numpy array but torch tensor
test_set = torch.FloatTensor(test_set)  # Warning: after running this the training set and test set on the Variable Explorer will disappear because Variable Explorer and Spyder does not recognize the torch tensors since py torch is so recent (couple of weeks). but don't worry, variables will be exist but it won't be shown on the Variable Explorer # Up until now, it was data preprocessing which is same in Boltzmann machine and autoencoder. from now on we code for the Boltmann machine which is, to predict if the user like the movies or not

# Creating the architecture of the neural network
class SAE(nn.Module): # SEA stand for stacked auto encoder. # Inside we put the parent class which is modulo. # you can see other parent classes by pressing the Tab bottom # this action is called inheritance
    def __init__(self, ): # init stans for initializating # after self we put a comma and then put nothing because this will consider the variables of the modulo of the class since we are doing inheritence
        super(SAE, self).__init__() # super function is used we want to use the inherited methods and classes from the modulo class # arguments: 1- SAE class 2- object of the SAE class which is self # .__init__() is used just to make sure that all of the inheritent classes and method of the parent class and modulo 
        self.fc1 = nn.Linear(nb_movies, 20) # Full connection between the first input vectore of features and the first encoded vector. we call the first full connection fcl # inputs: 1- Number of features in the input vector and since the features are the movies 2- Number of nodes and neurons in the fist hidden layer that is the number of elements first encoded vector. in here we put 20 by expriment but it has room for improvement.
        self.fc2 = nn.Linear(20, 10) # since we are making a stacked auto encoders then we put another layer which we are going to make the (second) full connection between first hidden layer composed of 20 neurons. # inputs: 1- Number of features in the input vector which is the number of first hidden nodes or 20.  2- Number of nodes and neurons  which based on expriment we choose 10. 
        self.fc3 = nn.Linear(10, 20) # Adding the third layer # inputs: 1- number of inputs which the number of previous hidden nodes 2- since in here we doing decoding and not encoding anymore which we are trying to reconstruct the original input vector which is 20 in other word, by symmetric the value is 20
        self.fc4 = nn.Linear(20, nb_movies) # in here since it's the output then the number of the second argument should be same as the number of inputs which is the number of movies
        self.activation = nn.Sigmoid() # defining teh activation function # By expriment between the Sigmoid and the Rectifier activation function. the Sigmoid will give a better result 
    def forward(self, x): # This is a function that does the action (encoding and decoding) that takes place in auto encoder. we use 'forward' for doing so. forward is like forward propagation. # Arguments: 1- self 2- input vector which is x
        x = self.activation(self.fc1(x)) # Firs encoding takes place here that is to encode the the input vector of features (x) into a vector of 20 elements # We use activation in here becuase it will activate the neurons of the first encoded vector that is 20. inside this activation we put our input vector. # This activation is same as the one created above # Arguments: 1- input vector which so far represented by x. here, this should be applied on the first full connection which is represented of fc1. so in order to include the the first full connection we don't input x directly. we put x inside the object fc1. # The output of this is the encoded vector
        x = self.activation(self.fc2(x)) # Doing the same thing for the second full connection
        x = self.activation(self.fc3(x)) # Note: in fc3 we are decoding and not encoding anymore
        x = self.fc4(x) # The activation function is not applied on the final part of decoding # This is the vector of predicted ratings and this is the vector that we will compare to the real ratings.
        return x # Final step
    
sae = SAE() # creating an object of the previous class # There is no need to put any inputs since all of them are previously defined
criterion = nn.MSELoss()  # Creating loss function  which is mean squared error   
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # Creating optimizer. opptimizer applies the stochastic gradient descent to update the weights in order to reduce the error at each epoch # By exprimenting between the Adam optimizer and the RMS prop class, the second one returned a better result. # Arguments: 1- All the parameters of our auto encoders which are the fc's and the sigmoid activation dunction. 2- Learning rate. by exprimenting 0.01 seems reasonably good 3- decay which is to reduce the learning rate after each efew pochs and that's in order to regulate the convergence. again the number is based on exprimenting

# Training the SAE
nb_epochs = 200 # by exprimenting
for epoch in range(1, nb_epochs+1):  # looping over epochs # nb_epoch+1 because upper bound is excluded.
    train_loss = 0 # 0 because at the start there is no error but during time it will be added
    s = 0. # Counting the nunmber of users that rated al least 1 movie # This for the purpose of optimizing the memmory because we won't do the computation for the users didn't rate any movie for doing so we keep track of them # The reason for making this float is because we are going to use s to compute the RMS at the end, that is the root mean squared error which is float then all of its computet elements should be float 
    for id_user in range(nb_users): # looping over all of the observations (users) in 1 epoch # here we don't start the range from 1, we do it from 0 because it needs to be the indexes of the observations in the training_set 
        input = Variable(training_set[id_user]).unsqueeze(0) # input vector of features that contains all the ratings of all movies given by this particular user # Here we add a additional (fake) dimension which corresponce to the batch. for doing so first we add Variable and then put the unsqueeze to 0 (0 is the index of our new dimension)
        target = input.clone() # Creating target # basiclly target is same as the input vector but since the input is going to modified then we like to get the original input before modification. we doing so by cloning the input
        if torch.sum(target.data > 0) > 0: # if condition to optimize the memmory which is to eliminate the observations that only contains zeros # target.data is going to be all the rating sof this users
            output = sae(input) # output vector of predicted ratings
            target.require_grad = False # This step is also about optimizing the memory. when we apply stochastic gradiant descent, we want to make sure that gradient is computed only with respect to the inputs and not the target.
            output[target == 0] = 0 # This step is also about computation. in future of computation of gradient and stochastic gradient descent, we only want to include the non-zero values in computation. so we don't want to deal with 0 ratings or no rating at all. but that's only for the output vector. # Read the [] as 'such as' # When we put zero for output to, it means that these values will not count in the computation of errors so they won't have the impact on update of diffrent weights 
            loss = criterion(output, target) # Computing loss function # arguments: 1- vector of real ratings 2-  vector of predicted ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # This represents the average of error by only considering the movies that have been already rated (between 1 - 5) # This is just to adapt to the 'if else' statement for consideration of non-zero ratings # the reason for adding e is becuase of the mathematical reason that we don't want the denominator be zero because it's not defined
            loss.backward() # Backward method for the loss
            train_loss += np.sqrt(loss.data[0] * mean_corrector) # Computing the RMSE and update the train_loss # after loss we put a .data[0] in order to access to a part of the loss function, that contains the error # Afteward we gonna multiply it by mean_corrector in order to adjust to it (non-zero ratings) # since the at the moment we have the squared error the at the end we take the square root 
            s += 1. # incrementing s in order to keep track of the users
            optimizer.step() # Updating the optimizers
    print('epoch: '+ str(epoch) + ' loss: ' + str(train_loss/s)) # The reason for diving over s is because we want the averege # this loss is for the training set
            
            
# Testing the SAE
test_loss = 0 
s = 0. 
for id_user in range(nb_users): 
    input = Variable(training_set[id_user]).unsqueeze(0) # In here we keep tht training set and will not change it to the test set. we do this only for the target. # training set is contain all the ratings of the movies by this specific user. then we predict the ratings of other movies that user hasn't watch yet. then in fututre we have the test set that contains the real answers that contains the real ratings for these movies that were not part of the training set. then in this way we calculate the loss.
    target = Variable(training_set[id_user]) # here we take the test set
    if torch.sum(target.data > 0) > 0: 
        output = sae(input) 
        target.require_grad = False 
        output[target == 0] = 0 
        loss = criterion(output, target) 
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) 
        test_loss += np.sqrt(loss.data[0] * mean_corrector)  # loss.backward() and optimizer has been eliminated in testing
        s += 1.
print('test loss: ' + str(test_loss/s))
          