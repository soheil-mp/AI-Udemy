
# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper # Importing all the tools and enviroments
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete # Importing the doom related packages codes which is its enviroment and the action it can play (which is move left, right, turn left, right, move forward, and shoot)

# Importing the other Python files
import experience_replay, image_preprocessing


# Part 1 - Building the AI            # We will implementing this in three parts: implementing the brain (neural network which in here it's CNN. in here it detects the images and prdict the Q-values) and body (specifying to AI to how play the action) and AI which is assembed of these two.

# Making the brain
class CNN(nn.Module):
    
    def __init__(self, number_actions): # arguments: 1- self 2- number of actions in the doom enviroment (diffrent enviroment has diffrent number of actions)
        super(CNN, self).__init__() # activating the inheritence and using alll tools in nn.Module
        self.convolution1 = nn.Conv2d(in_channels =  1, out_channels = 32, kernel_size = 5) # architect: first  we are going to build a CNN with three convolutional layers and one hidden layer afterward. it means that we need 3 convolutional connection and two full connections # we are going to flettening the all the pixels into a vector that becomes the input of the neural network # arguments: 1- in_channels: input of the convolution which is the number of channels in our image. since we are going to work with black and white image then we put 1 because the AI is totally capable of recognizing the monsters in black and white. 2- out_channels: output of the convolution which is equal to the number of features we want to detect in our original images. comman practice is to start with 32 featuer detector. this 32 means that after that the concolution has been applied to the input image, we get 32 new image with detected features 3- kernel_size: dimension of the square that will go throught the original image. common practice is to use either 2x2 or 3x3 or 4x4 or 5x5. in here we start with 5x5 (because we want to detecting the big features.) an then reduce it for the next convolutional layers. this is like when seeing someone, first you detect the big picture like their feature and then you dect smaller details like their eyes, smile, etc. 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # input channel in here is the output channel of the previous layer
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) # sine in here we are detecting smaller features then it makes to use detect more of them so we need bigger output channels # After these three covolutional layers we take thie flattening which is 64 x 32 x 32 images that we got from these convolutions and then input it to fuly connected neural network
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), # number_neurons has been replaced
                             out_features = 40) # first full connection is from the fletten vector to the first hidden layer and the second one is between this hidden layer to output layer composed of output neurons that are the Q-values for possible actions # arguments: 1- in_features: number of the pixels in the huge vector optained after flattening all the process images after the 3 convolutions. since this number is hard to get then we use a function for it. we name it number_neurans and we make a function for it afterward. 2- out_features: the number of neurons in hidden layer. we start with 40 and we can do further exprimenting for it.
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions) # arguments: 1- in_features which is the out_features in the previous fc 2- out_features: number of output neurons and since each output neuron correspond to the new Q-value and one Q_value correspond to one action. so number of output in here is the number of actions which is number_actions as defined in the argument of __init__ function.
        
    def count_neurons(self, image_dim): # arguments: 1- object which is self 2- image_dim: number of output neurons in the flattening layer depends only on one thing which is the dimension of the input image (the ones that goes at the very begining of the neural network) so the dimension of the input images are the only argument we need.  # The actual dimension of the input images (coming from doom) are going to be 80x80. we're going to reduce the size of the original images to 80 x 80. that's going to be the format of the images that is going to the neural network. so image_dim is going to be 80x80 and also a 1 which correspond to black and white images that we are working
        x = Variable(torch.rand(1, image_dim)) # Since at the start we don't have any input images (doom images which we are going to import later) then we have to create a fake image (and also fake pixels) of 80x80 # rand is to make a random fake pixels and inside we put the dimesnsion of the images which is 1, 80x80 # arguments: 1- fake dimension: since we are going to put this image into the neural network, then we need a batch of input states (input images in here). so we are creating a fake dimension by puting 1 which 1 coresponce to the batch 2-  dimensions: we can also put the tuple (80, 80) which these dimensions are contained in image_dim argument. the reason for using star * before this is because to passing the elements of tuple as a list of argument of a function # at the end we have to convert the input vector into a torch variable because this is going to the neural network
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # since we need only the number of neurons, so after the convolutions has been applied then we will go just up to the convolution3. so we go only after this and we won't go into the two full connection. because number of neurons that we want is betweek convolution3 and fc1 # Propagating the input image into the neural network to reach the flattening layer then we are going to get the neurons in the flattening layer. note that we are only propagating in the concolutional layers until the flattening layer is reached. # This is a 3 step process: STEP 1: applying the convolution into the input images by "self.convolution1(x)", STEP 2: applying the max pooling to optain the convoluted images by "max_pool2dself" note that in here we have to put an additional argument inside this wich is: 1- self.convolution1(x) 2- the kernel_size: size of the window that slides through our images. we choose 3 which is common choice. 3- strides: by how many pixels it's going to slide in the images which 2 is common choice, step 3: activate the neurons in this pooled convoluted images by "F.relu"
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # Doing the same thing to the convolution2
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) # Doing the same thing to the convolution3
        return x.data.view(1, -1).size(1) # first flatten all the pixels of the third convolutional layer and then getting the number of neurons

    def forward(self, x): # propagating the signals in all the layers of the neural network. # arguments: 1- obejct which is self 2- x: which is the first the input images and then it will updated as the signals (as it propagated into the neural network)
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # these 3 lines are same as the one the we build previously.
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1) # propagating the signals from the convolutional layers to the hidden layers and eventually to the output layer (at the very end of neural network). for doing so, first we need to fletten the third convolutional layer # arguments 1- x.size(0) to take all the pixels of all the channelsand the third convolutional layer 2- -1 to get the end of it # this trick is used to flatten a convolution that composed of several channels by using the size function. for more detail check the PyTorch website.
        x = F.relu(self.fc1(x)) # after getting the flattening layer, it's going to be the input of fully connected network (fc1 which connects the flattening layer to the hidden layer) with a simple linear transmission of the signal with a linear class and the to break the linearity (because we are working with images and images have non-linear relationship) we use rectifier function to learn the non-linear relationships. 
        x = self.fc2(x) # propagating the signal from the hidden layer to the output layer with the final output neuron
        return x # now the process of making the brain is finished.
        
        
# Making the body       # defining how actions are going to be played which like the human body signals are gonna come from the human body. we do so with the softmax method. so CNN is the brain and SoftmaxBody is the body.

class SoftmaxBody(nn.Module): # the reason for no choosing just Softmax is because softmax is a class in pytorch and we don't want the intersections. 
    
    def __init__(self, T): # arguments: 1- object which is self  2- tempreture which is the same parameter as the one we used in self-driving car.
        super(SoftmaxBody, self).__init__()
        self.T = T # Setting the tempreture variable which is equal to the argument that will be input when creating an object at the class.

    def forward(self, outputs): # making the forward function to propagate the output singnals of our brain to the body of the AI so that it will play the right action to reach the vest. (there is no right action yet because there is no training, which we will do so in Part 2) # arguments: 1- object which is self 2- outputs which correspoce to the output signals of the brain
        probs = F.softmax(outputs * self.T) # first step: we will get out distribution of probabilities for each of the Q-values which depend on the input image and each action. we have one Q-value for each of the possible actions (around 6 or 7 in here) and therefor we get a distribution of seven probabilities # arguments of softamx: the elements for which you want to create a distribution of probabilities. that is the Q-values that is the outputs of the neural network. # the reason for creating this distribution of probability is to be able to explore diffrent actions instead of directly picking the one that has the maximum Q value (because of exploration and explotation). # multiplying with the tempreture parameter to configure the customize the exploration (higher tempreture = less exploration = action whith the higher probability is going to be chosen)
        actions = probs.multinomial() # second step: sample the final action according to this distribution of probabilities. we do so with the multinomial method
        return actions
       

# Making the AI   

class AI:   # Now that we have the brain and the body we are going to assemble them. 
    
    def __init__(self, brain, body): # arguments: 1- self  2- our brain  3- our body
        self.brain = brain # defining the variables of our AI object # the brain argument (right side) later will be the brain object created from the CNN class
        self.body = body # the body argument (right side) later will be the body object created from the SoftmaxBody class
        
    def __call__(self, inputs):  # now that we have our AI we need a big forward function called __call__ (we don't use our previous two forward function seperatly that will take the images as input then propagate the singlas into the brain with first forward function and once it gets the output signals of the brain it will forward this output signals into the body with the second forward function and then return the action to play) now we make the big forward that will take the input images at the begining and then return the action to play with the assmbled AI.  # arguments: 1- self  2- input images
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32))) # first step is to recieving the input images from the game. since these images are going to enter the neural network. then change the formats first from the images to the numpy arrays and then from numpy array to the torch tensor and from thr torch tensor to the torch variable that will contain both the tensor and the gradient. after getting the right format we enter it to the neural network and then we do the whole propagation of the signals.  # since the cells of the numpy array will contain the pixels, it is safer to use a float dtype. another reason for doing so is because tensors are arrays of single type so we choose the signle type to be float32
        output = self.brain(input) # propagating these allowed images into the eyes of the AI that is thorugh the 3 convolutional layers # output in here is the output signal of the brain
        actions = self.body(output) # propagating the output signal to the body and to do this we use the second forward function # actions in here is same as the action in SoftmaxBody
        return actions.data.numpy() # conveting back the torch format to the numpy array. we doing so by adding .data.numpy()
    
        
# Part 2 - Implementing Deep Convolutional Neural Network
        
# Getting the Doom enviroment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True) # PreprocessImage is to preprocess the images that will come to the neural network we preprocess the so the have a square format (width = 80, height = 80) if remmbered we set the input images in the neural network to the 80x80 # greyscle = 1 which means the grey images that we also set this to 1 in out neural network (1, 80, 80)  # (gym.make("ppaquette/DoomCorridor-v0")) is the name of the enviroment
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True) # Importing the whole game with the videos with this line of code.
number_actions = doom_env.action_space # if you remmber one of the arguments in our neural network was number_actions which is number of possible actions that the AI can do. now we want to get the number of actions in this doom enviroment # action_space is the sets actions (which works only in gym) and to acess the number of actions in the enviroment we add n which is the number of actions. # this will return 7 because there are 7 possible actions. there is 6 in the gym page but there is also running which will be 7.
    

# Building the AI               # so far we have only made the manual instruction with the AI class but we haven't created any object yet so we don't have any actual AI yet. this object will be nothing else than the AI that will have the brain and the body
cnn = CNN(number_actions) # cnn is the brain object # At firs we will create a brain object and body object and then building the actual AI # the reason for putting number_actions is because in the __init__ function of CNN we have so for creating the object we need to initialize it as well.
softmax_body = SoftmaxBody(T = 1.0) # making the body object # the reason for putting T is because in the __init__ function of SoftmaxBody we have so for creating the object we need to initialize it as well. since we have't initialized the T like the number_actions before then we will initialize it here to 1.
ai = AI(brain = cnn, body = softmax_body) # making the final AI which is the AI object


# Setting up the Exprience Replay                   # we already have a implemented version of exprience replay in the folder, beside it's adapted to eligiblity trace. the file has 2 classes, one makes the AI progress doing n steps so it can sum the rewards that have been observed on these n steps
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_steps = 10) # STEP 2# this is the steps for eligblity trace which is required for one of the arguments for making an object of ReplayMemory # arguments: 1- enviroment which is the doom enviroment that we have imported # AI which is the ai object we have build 3- n steps that we want to learn each 10 steps
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000) # STEP 1  making an object of replay memory class # arguments: 1- number of steps which corresponce to the number of steps which we'te going to learn the Q-values. that is the number of steps which we accumelate the target and the reward. we are going to have the acumulative target and acumulateive reward instead of single target and reward. with this technique the training time is going to be decreased 2- Capacity: size of the memory which we get the memory of last x steps performed by the AI. # before we only had the exprience replay. now that we have also eligblity of 10 steps, the training performace is going be improved.


# Implementing Eligiblity Trace                     # This algorithm has been taken from the 'Asynchronous method for deep reinforcement learning'. in here we build the algorithm for 'Asynchronous n-step Q-learning' with only one agent so we won't say asynchronous in here therefor we call it 'n step Q-learning eligblity trace' or even Sarsour, it's in page 4 and pscodocode in page 13 (line 18 from 'until'). the algorithm for A3C is also in this paper. there is only one diffrence in here that we use softmax method instead of e-greedy but the rest is same. 
def eligibility_trace(batch):       # in here we don't need to build a class, function is enough because we don't need a object for this. because basiclly we need to return the inputs and targets so that later when training the AI we are ready to minimize the distance between the predictions and the target. to get the prediction, we are going to apply the brain to the inputs to get the output singnals that will be our prediction. once we have the prediction and the target we are ready to train the AI by trying to minimize the square distance between the predictions and the targets # argument: 1- batch: in here we only need batch as out argument because we are going to get some inputs and some targets because we are going to train the AI on the batches so the inputs and the target will go inside the batches.
    gamma = 0.99 # first we need a gamma parameter  # 0.99 is a clasic goog value for the gamma # next step is to prepare our inputs and targets whic we will initialize them with an empty list
    inputs = []
    targets = []
    for series in batch:  # Computing the acumalative reward over the 10 steps. in each step we are goin to get the maximum of the Q-values of the current state that we are during this n steps we are in. if we reach to the last state it will be 0 because we don't want to update it anymore # following the instructions in the paper. REPEAT in pscodocoe means 'for loop'. for means the second 'for loop' # in the second for loop, we will update the reward in this way: by multiplying it by the decay parameter gamma and adding the reward # in here the iterative variable is going to be the 10-steps series (series of the 10 transition) and we call it series that represents the series of the 10 transition which we will consider them in our batch
        input =  Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32))) # For getting the acumalative reward, we need the state of first and last transition of the series. so now we are going to get these input states. # first we put these input states into a numpy array # for getting the first input state of first transition, we write series[0] and for accessing it we can write down .state (which we defiend it in the exprience replay). for acessing the last state of transition only replace the 0 with -1 # after this but this into a list because that's what the numpy array expecting # we want to choose the float32 type in here becuase later on we put this into torch tensor
        output = cnn(input) # now that we have the inputs we will get the output signals of the brain of the AI which is the prediction.
        cumul_reward = 0.0 if series[-1].done else output[1].done.max() # Computing the accumalative reward (which is R in the paper). at each step of the 10 steps run we need to update it by  adding a 0 to this accumalative reward if we reach to the last state of the series. or maximum of the Q-values if we haven't reach the last ste of series. # done is and attribute of the transition structure that we defined in the exprience replay. it means that the transition or step is over. (look at the gym documentation in the Doom part for codes) # output[1] because we want the (maximum) Q-values and they are in the incdex 1
        for step in reversed(series[:-1]): # for each 10 steps updating the ccumalative reward. by multiplying it first by gamma and then adding the reward. in the soducode the for loop starts at the end ant goes to the begining  # the trick to go from right to left in the list to use reversed # series[:-1] becuse if you check out the paper, the for loops starts at t-1 and by puting [:-1] means that we are going up to the element before the last element but not up to the last element.
            cumul_reward = step.reward + gamma * cumul_reward # based on the paper we multiply it by gamma and then add it by the reward optaind by the current state (that is in the step of the for loop) # reward is an attribute of step object 
            state = series[0].state   # Now we want to make the inputs and the target ready. the first thing to do is to add the firs state in the inputs list. so we are goin to get the first input state separetly. we can get it by copy pasing from the above.
            target = output[0].data # getting the targets seperatly associate to the first input state of the transition. target in here is equal to the Q-values of the first step # the reason for the output[0] is beacuse Q-value is return by he neural network which is contained in the output and since output is associete to the input then we can get it by the 0 index. that is because the Q-value of the input state of first transition and that is exactly the target Q-value.
            target[series[0].action] =  cumul_reward # updating the target variable but only for the actions that were selected in the first steps of the series (which for accessing them we need series[0])  # .action is to access the actions corrspondes to the first step of the series. # note: each transition of series has the following structure: state, action, reward and done # the target for that specific action of the first step is exactly what needs to be updated by the acumulative reward.
            inputs.append(state) # updating the input by appending the first input state and the first target from the above # we only need to update the first step of series because we train the AI on the n steps and therefor input is the fisrt step of the ten steps and also we get the target in the first step but we don't get an inputs or any targets in the following steps of the 1o steps because learning happens 10 steps afterward (that's why we get the state and the target of the first step of series)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)  # returning the inputs and targets in numpy array first and then in torch. # we are going to us a trick for the traget whici is quicker which we are going to stack the target together.
            

# Making the moving averege on 100 steps
class MA:  # before we doing the training, we are going to make a class for making the moving averege for keep tarcking of the averege during the training
    
    def __init__(self, size):  # arguments: 1- object which is self 2- size which correspond to the size of the list of the rewards of which we are going to compute the averege so this is going to be 100
        self.list_of_rewards = []  # initializing the variables specific to the object # this is foing to be the list of 100 rewards we are going to compute the averege
        self.size = size # this is going to be equal to the argument of this function
    
    def add(self, rewards): # this will add the cumulative reward (not the simple reward because we are doing the eligiblity trace for learning each 10 steps) to the list of rewards # arguments: 1- self 2- rewards which represent the cumulative reward and this is a list
        if isinstance(rewards, list): # when we get a new cumulative reward (after progressing on 10 new steps) we have to add these cumulative rewards to the list. for doing so we have to seperate two consitions (since we are working with batches while reward be in lists but in some other cases rewars can be as a single element which in this case the syntax for adding an element to a list which is list of rewards is not same as adding a single element). so in here we have to make this condition that seperates these conditions. # this line means that "if the rewards are into a list"
            self.list_of_rewards += rewards
        else: # if the 'rewards' is not a list but it's a single element so we should use a "append function"
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size: # now we have to specify that what will happen when the list gets more than 100 elemens which in this case we have to delete the first element of the lsit to make sure that we always have 100 elements # self.size will be initializez later since it's out argument
            del self.list_of_rewards[0]
    def averege(self): # Computing the averege of list of rewards will contain the 100 elemnts
        return np.mean(self.list_of_rewards)
    
ma = MA(100) # making an object of moving averege (MA) class and we want the size of 100 becuase we want to compute the moving averge on 100 steps
            

# Training the AI
loss = nn.MSELoss() # since we are predicting the Q-values of diffrent actions and these are the real numbers then we are doing some neural network for the regression then the loss function is the mean square error
optimizer = optim.Adam(cnn.parameters(), lr = 0.001) # arguments: 1- parameters: this is the one that make the connection between the optimizer and parameters of our neural network (that is the wights of the neurons in our brain) for doing so we take our brain which is cnn. 2- learning rate: in here we want smaller learning rate because we don't want to converge too fast and we want to have so exploration which a good value is 0.001
nb_epochs = 100
for epoch in range(1, nb_epochs + 1): # we want the range of nb_epochs which in here the upper bound doesn't count
    memory.run_steps(200) # now we want to do 200 runs of 10 steps. so each epoch will be 200 runs of 10 steps. for doing so we have the run step function from exprience replay class. we will get it from the memory object (of replay memmory class) to generate these 200 runs of 10 steps
    for batch in memory.sample_batch(128): # sampling some batches from these runs. for doing so we have another function from our memory which is sample batch which it will generate some batches from these 200 runs. note that this time, these batches are the batches of series of transitions that is the series of 10 steps. (before it was just the batches of single transitions which in here is 10 steps). for doing so we use sample_batch the input is the batch size which we can take 32, 64 or 128. # common sense is to start 32 but since we are doing the 10 steps in here then it's better to take the batches with the larger size (64 or 128).  # after this we put this inside a for loop because we are taking serveral batches and we are taking them in what's return by the sample batch function. # this for loop menas that every 128 steps our memory will give us a batch size of 128 which will contain the last 128 steps that were just run. the learning is going to happen on these batches. also inside these batches we are going to have eligiblity trace to learn on rach 10 steps.
        inputs, targets =  eligibility_trace(batch) # getting the inputs and tragets seperately. we will do so with one of the tools that we have implemented for eligiblity_trace  # this batch is the batch for our for loop
        inputs, targets = Variable(inputs), Variable(targets) # converting to the torch variable
        predictions = cnn(inputs)  # getting the prediction
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()  # initializing the optimizer
        loss_error.backward() # back prpagation
        optimizer.step() # Updating the weights
    reward_steps = n_steps.reward_steps() # computing the cumulative reward which we can do with n_steps objects (of reward_steps in exprience_replay.py) from above which will get the cumulative rewards happening during the n steps run. # we use this to first update the new rewards of the steps. then updating the moving averege object by adding th cumulative rewards to it and recomputing the averege # the output in here is the new cumulative rewards of the steps
    ma.add(reward_steps) # adding the the new cumulative rewards to out moving averege object. for doing so we have a method in the moving averege class (MA) which is the 'add' method.
    avg_reward = ma.averge( # computing the averge reward. for doing so we use the 'averge' method in MA
    print("Epoch: %s, Averege Reward: %s" % (str(epoch), str(avg_reward)))  # printing the averege reward every epoch. we want to see the increasing averge reward. # at start we have the exploration phases so it might not be a big increasing at start. it only increases on the explotation phase. # %s is to convert everything to the string
    if avg_reward >= 1500: # by exprience for sure it will reaches to vest by 1500
        print("Congradulation, your AI wins!")
        break # stop the training and the whole process if the 'if statement' is true
        
# Closing the Doom enviroment
doom_env.close()



