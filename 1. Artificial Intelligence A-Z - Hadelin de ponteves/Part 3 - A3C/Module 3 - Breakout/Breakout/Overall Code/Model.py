# in the model.py we build the brain of whole A3C model. in here we build some neural network and also convolutional neural network because we still doing some deep reinforcement learning. also inside this neural network we integrate everything related to actor critic model. what makes this more poweful is that it's also contain LSTM so we can learn the temproral properties of inputs so that the predictions can be better

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F             

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std = 1.0):    # Start by making two functions (this and the blow function) that help to initialize the weights.  # arguments: 1- initializing the weights 2- initializing the standard deviation because we want to set a specific variance for our tensor of weights. the reason for doing so is becuase when we make the neural network, there will be actor and critic and we make two fully coneccted layers, one for the actor and one for the critic. these two fully connected layers will have weights and we will set a standard deviation for each these two groups. so we will set a small standard deviation for the actor (around 0.01) and large standard deviation for the critic (around 1). # inside the paranthesis we put a defualt value of 1 which later can be change. 
    out = torch.randn(weights.size()) # first we prepare the output which this will be the tensor of weights (random weights that follow a noraml ditribution with randn which n stands for noraml) that will have a specific standard deviation. # the input is the number of elements that distance will contain which is equal to the number of weights becuase we are initializing a tenosr for these weights in here and for getting the number of elements use .size() afterward we are going to have a torch tensor that have the same number of element of weights and it will be initialized with random weights following the normal distribution.
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # setting the standard deviation which in here we will do the simple normalization # in sqrt we input the sum of the square weights of vector. inside the sum we input the the index that contans the weight we want to sum # the reason for using expand_as() is becuase we to get these weights seperately so we want to sum them up. # so expand_as will get the weights of 'out' which was initialized as the torch tensor of weights afterward we get the sum of squares of the weights and at then end we take the square root to apply the normalization # the fact that we have std in our numerator will make sure that var(out) = std^2 
    return out

# Initializing the weights of the neural network for an optimal learning
def weights_init():  # arguments: 1- the object m which will represent the neural network.
    