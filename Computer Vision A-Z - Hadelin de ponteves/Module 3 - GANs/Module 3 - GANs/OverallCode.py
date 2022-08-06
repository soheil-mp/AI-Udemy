
# importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some Hyperparameters
batchSize = 64 # Setting the size of the batch
imageSize = 64 # Setting the size of the generated mages (64x64)

# Creating the transformation
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images. # this transformation is for the generator

# Loading the dataset
dataset = dset.CIFAR10(root='./data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers= 2) # We use dataLoader to get the images of the training set batch by batch. # shuffle is to get the images in a random order # num_workers = 2 means that there is going to be 2 parrallel threads that will load the data # This approch of getting the dataset and also num_workers make the process much more faster when the dataset is huge

# Defining the weight_init funcntion that takes as input a neural network m and that will initialize all its weights
def weights_init(m): # This is for both generator and discriminator 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# Defining the generator      # We do this on two step: 1- defining the architecture of the neural network of the generator 2- Making a forward function to propagate the signal inside the neural network 3- Creating the generator itself by creating an object of the class
class G(nn.Module):  # G stands for generator # in here we use trick in object oriented programming which is inheritence which here, it's to inherit from this modulo that contains all the tools that allow us to build a neural network for example one modulo can be convolution, another modulo can be full connection, etc.
    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__() # Now we are going to activate the inherihance by super function which we put our class G inside # self is because we use the tools of that modulo in out project # after super we just write init to activat the inheritance and it's not so important
        self.main = nn.Sequential(    # Making a meta modulo (a huge modulo that is composed of several modulo). here the modulo means the diffrent layers, diffrent connection inside the neural network # main is the meta modulo that contains diffrent modulos in a sequence of layers # Sequential is the sequence of layers and inside we build diffrent modulos # self.main is an object and the Sequential is a class
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # First modulo: Inverse of convolution # The reason for starting with this is because the role of generator is to generate some fake images. Since CNN takes images and return the output vector (We call it noise), the inverse CNN will do the exact opposite which takes noise and outputs image. we do it for a random vector of size 100. # ConvTranspose2d() is the inverse convolution # ConvTranspose2d: 1- Size of the input 2- Number of feature maps of the output 3- Size of the kernel that is 4 which it means the kernel will be the squares of size 4x4.  4- Stride 5- padding 5- bias which by default is True but we want it False because we don't want bias # The choice of number came from research papers that they were exprimenting on lots of numbers.                                                                                                                       
            nn.BatchNorm2d(512), # Normalizing all the features along the dimension of the batch which here is 512 # arguments in here are the number of the feature maps we want to batchNorm # We want the same 512 feature maps
            nn.ReLU(True), # Applying ReLu rectification function to break the linearity for the non-linearity of the neural network # As its argument just put True
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # Applying another inverse convolution # the input is not 100 anymore. it's the output of the previous one which was the feature maps (512) # Now the output is 256 which is the number that is find by reasearching # in here the Stride and padding has been change
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Applying another inverse convolution
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # This is the last inverse concolution # since in here we make the generator, generate som fake images and since these fake images are going to be in 3 channels then the number of dimension for the output is going to 3
            nn.Tanh()) # At the end we are applying the hyperbolic tangent rectification to 1- break the linearity 2- to make sure that we are between -1 and +1 and is centerd on 0. the reason for doing so is becuase to have the same standards as the images of the dataset and because the created images of the generator will become the input of the discriminator
                
    def forward(self, input): # Forward propagating  # input which is the input of the neural network of the generator. this input is going to be some random vector of size 100 that's why we also specified the 100 in inverse convolution. this is just some random vector that represend some noise to generate fake image that will be the output of the generator. 
        output = self.main(input) # as an argument we put the input and this will returns the output
        return output 
    
# Creating the generator
netG = G()   # Now we can make as many generator as we want by creating an object of the class G # netG stands for the neural network of the generator
netG.apply(weights_init)  # Initializing the weights proper way which weight_init function will do it for us. # if you check out the weight_init funciton you will notice that there is 'Conv' and 'BatchNorm' which Conv will initialize the weights (0.0, 0.02) to the ConvTranspose2d (since Conv is part of this) and BatchNorm will initialize (1.0, 0.02) to the BatchNorm2d (becuse 'BatchNorm' is part of the name of this). also in each BatchNorm we are going to have some bias that we initialized it to 0 in above.
        
# Defining the discriminiator      
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__() 
        self.main = nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), # The discriminiator takes as input a generated image from the generator and returns as a output a discriminiating number which will be a value between 0 and 1. so this time we use Convolution and not the Invese Convolution because it takes image and not a value. # Arguments: 1- in_channels: dimension of the inputs. for choosing this number, we should look the output of the generator which is 3.  2- out_channels: 64 feature maps 3- kernel_size which is 4 which is going to be a square by the size of four by four  4- Stride which is two  5- Padding which is one 5- Bias which we don't want to have 
            nn.LeakyReLU(0.2, inplace=True), # the BatchNorm is going to be applied from the next convolution. however the rectification going to be applied. in here, insted of "ReLU", the "leaky ReLU" is going to be used. again all of these choices come from the reasercher in which they exprimented with each of these. # aguments: 1- negative slope: the graph contain a negative slope from the ReLU in which, we choose 0.2 which come from exprimentation 2- inplce: this should be True
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), # The reason why the output is 1 in here, is because the discrimiator is returning a discrimiating number between 0 and 1. so it's a simple vector of one dimension containing this number 
            nn.Sigmoid()) # Sigmoid is the best choise since it returns a value beteween 0 and 1 # in discriminiator 0 corresponce to the rejection of the image and 1 corresponse to the acceptance of the image. by using the threshold tecnique which if the value is below 0.5 it return 0 and if above it return 1
                
    def forward(self, input): # There is no dangeur to call this forward again
        output = self.main(input) # The discrimiator takes a inputs an image created by the generator and for this image, it will decide whether it will actepted (by 1) or it will rejected (by 0) so the output is the discrimiating number between 0 and 1 (which afterwrard it will evaluated with help of threshold)
        return output.view(-1)  # There is a little trick that we should choose now. since in the architecture of the neural network of discriminator, it is sequence of the convolutions. so at the end of CNN we need to flatten the resualt of all convolutions. view(-1) will do the exact thing
    
# Creating the discriminiator
netD = D() 
netD.apply(weights_init)  
        
# Training the DCGANs               # This is in two steps: 1- Updating the weights of the neural network of the discriminator. this has some sub-steps as follows: 1.1- Train it by giving it a real images (to see whats real and what's not) and setting the target to 1.   1.2- Train it by giving it fake images (generated by generator) and setting the target to 0.    2- Updating the weights of the neural network of the discriminator. this has some sub-steps as follows: 2.1: Take the fake images in one step of the loop and then feed it to the discriminator to get the output (between 0 and 1)  2.2: Seting the new target to 1   2.3: computing the loss between the output of the discrimiator (value between 0 and 1) and this target which is equal to 1   2.4 back propage this error to the first neural network which is generator
criterion = nn.BCELoss() # measuting the error of the prediction (discriminator between 0 and 1) and a ground truth that is either 0 (fake) or it's 1 (real). # BCELoss (Binary Cross Entropy) is a formula that you can cheack it out on the documentaion site of PyTorch. this is also used for auto encoders.
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas= (0.5, 0.999))  # One optimizer for generator and one for the discrimiator # Optim library has been imported # Adam is the higly advanced optimizer stochastic gradient descent. # Arguments: 1- parameters of neural network in discrimiator 2- learning rate which is 0.0002 3- beta's parameter which the numbers come from exprimenting. to know more about check its website
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas= (0.5, 0.999))

for epoch in range(25): # there is going to be 25 epochs. # you can change this for having diffrent results but this is good too.
   
    for i, data in enumerate(dataloader, 0): # Going through all of the images in the dataset # data is in which contains the mini batch of the images # this means that we go though all of the images of the dataset mini batch by mini batch # second argument is where the index of the loop is going to start which is going to be 1
        
        
        # Step 1: Updating the weights of the neural network of the discriminator
        netD.zero_grad()  # Initializing the gradient of the discriminator with respect to the weight to zero and that's what the zero_grad does
        
        
        # Training the discriminiator with a real image of the dataset
        real, _ = data # Getting the mini batches of real images. this is going to be the first element of our mini batch data # At the moment we are dealing with the specific mini batch which is our data and data is composed of two elements. first element is the real images itself and second element is the labels. in here we only care about the first element. # Our input is not accepted in PyTorch yet. it only accepts the inputs in torch variables. so...
        input = Variable(real) # Converting to torch variable
        target = Variable(torch.ones(input.size()[0])) # There is two diffrent target for the training. there is 1 which is for the real images and there is 0 which is for the fake images. so in here we specify that the ground truth is 1. # Now we are going to create a torch tensor that is going the have the size of the mini batch and that is going to composed of only ones. there is going to be ones for each of the input image of the mini batch. # inside paranthesis, the size of the torch have to be written that is the first index of the input.size() # Next step is to wrap this in a torch variable.
        output = netD(input) # Getting the output. since already have the input and the target. # The out is the neural network of discriminiator that as an argument we put our input. # The output is a number between 0 and 1 for acceptance or rejection
        errD_real = criterion(output, target) # Getting the error. The first error that comes from the first training of the discriminiator with real image ground truth. # errD_real since it coorensponce to the real grounf truth. # criterion calculates the loss function (between the output and target). inside the paranthesis, first we put the output and then the target.
        
        
        # Training the discriminiator with a fake image generated by the generator         # To train the discrimiator to see and underestand what is fake that is to recognize the fake images.
        noise = Variable(torch.randn((input.size()[0]), 100, 1, 1)) # creating a random vector of size 100 (if you notice in above we had the same size in our generator) that represents some noise and then feeding it to the neural network of the generator. at the start the images look like nothing but over the process of the updating the weights the image is going to be more like the real image. # The arguments of randn: 1- Batch size which is 64. (here we are not going to create one vector of size 100 but creating a mini batch of random vector of size 100) 2- Number of elements we want in this vector that is 100 because we specifiy the same thing in the architecture of the generator 3,4: we put ones in here to to give some fake dimensions to this random vectors that will coorespoce tot he feature maps. this means that instead of having 100 values in the vector, we have 100 feture maps of size one by one # At the end we wrap this baby with a torch variable
        fake = netG(noise) # fake represend the mini batch of the fake images so fake is the name of mini batch. # This is the new ground truth of the fake images
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach()) # value between 0 and 1 # in order to speed of the computing speed we want to detach the gradient from the 'fake' since we don't need them in part of stochastic gradient desent and it's going to be an extra computation # fake is a torch variable and it contains not only the tensor of prediction of the discriminiating numbers between 0 and 1 and also the gradient. we are not going to use the gradient after back propagating the error back inside the neural network when applying stochastic gradient desent
        errD_fake = criterion(output, target)
        
        
        # Backpropagating the error
        errD = errD_real + errD_fake  # Getting the total error and then backpropagating this error back to the neural network of discriminiator to then update the weights through the stochastic gradient descent according to how much they are responsible for the error
        errD.backward() # backward() is used to back propagate to the network  
        optimizerD.step() # Applying stochastic gradient descent to update the weights by taking the optimizer of the discriminator (from above) and then using step function in which it applies the optimizer on the neural network of the discrimiator to update the weights


        # Step 2: Updating the weights of the neural network of the generator              # This time the target is goin to be 1 eventhough this time, the image that will be the input of the discrimiator will be the fake image of the generator
        netG.zero_grad() # Initializing the gradient of the generator with respect to weights to zero
        target = Variable(torch.ones(input.size()[0]))  # Since we have already the inputs of the fake eimages of the mini batch and therefor we are going to get the target directly # Here we get the target of ones
        output = netD(fake)  # Output of discriminiator when the input is a fake image # Getting a value between 0 and 1 # This time we are not going to use the detach because we want to keep the gradient of fake because we want to update the weights
        errG = criterion(output, target) # Getting the error of the prediction (related to the generator) because this time we back propage this error to the neural network of the generator
        errG.backward()
        optimizerG.step() # Using the optimizer of the generator to make sure that this time, it's going to be the weights og the generator that is going to be updated. # The final result will be appear in the results folder.
        
        
        # Step 3: Printing the losses and saving the real images and the generated images
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) # First bracket: First %d corresponse to the epoch the second %d corrensponce to the 25 in (for epoch in range(25)) # Second bracket: doing something similar for the steps. firs %d coresponce to the step 1 or i. second %d corresnpoce to the number of elements in dataloader in (for i, data in enumerate(dataloader)) # printing the Loss of discriminiator after each step of the each epoch # %4 is to add some percentage with four decimals # The same thing for generator as well # after the quotes add another % and then add the elements you want to put in. # errD is for the Loss_D. errD.data[0] will get exactly what we want which is the value of the error of the discriminator
        if i % 100 == 0: # Using if else to save the real images and generated images of mini batch every 100 step # This is the rest of the division of the i to 100 is equal to 0 
            vutils.save_image(real, '%s/real_samples.png' % './results', normalize = True)  # vutils is imported from above which, it allows us to save the real images that the model was trained # arguments: 1- batch of rel images which we called real 2- name of the path leading to the location where we want to save real images. % s refres to the root (results folder in here). after %s is going to the name of the png file 3- Normalize which we want to be true
            fake = netG(noise)  # to get the fake images by calling the netG again
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./results', epoch), normalize = True) # %03 is the number of the epoch when the fake images are saved in 3 integers