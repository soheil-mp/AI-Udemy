
# Do the data preprocessing manually in the folders. note that feature scaling is 100% compulsory in deep learning (and especially in computer vision) but since we don't use the data preprocessing like before we will take care of that just before we fit our CNN to our images.

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # To initialize the neural network. there are two ways for initializing: 1. sequence of layers 2. graph. for CNN, it is the first one
from keras.layers import Convolution2D # To add convolutional layers. this images are in 2D but videos are in 3D
from keras.layers import MaxPooling2D # To add pooling layers
from keras.layers import Flatten # To convert the pooled feature maps into a feature vector
from keras.layers import Dense # To add a fully connected layers and a classic artificial neural network

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))   #filters: the number of filters or feature detectors and also to get the same number of feature maps. the other name of feature detector or filter is also convolution kernel. it has 3 names. the defualt for filter is 64 but we don't use this first. common practice is to start with 32 in first convolutional layer and then we add other convolutional layer with more feature detector like 64 and then 128 and then 256 # two 3's refer to the three by three dimention. so far we have created 32 feature detector of three by three dimention # input_shape converts all images into a single format and fixed size. we will actually do this convertion after we build our CNN and before we fit CNN into images. but in here we write what is the expected format of our input images. in input_shape we have three numbers: 1. number of channels so as in noted it's 1 if it's black and white and it's 3 if it's color // 2. & .3: the dimentions of 2D array into each of the channel. in here it means that we are expecting a collored image of __ times __ pixels. in here we use 64 because we using CPU so it's enough but in GPU we using more pixels for example 128 or 256. the more pixels the more time it will take. one last important thing that this ordering is in Theano but in thensorflow we use the dimention at the end. # The reason we used rectifier in here is because we seeking for the non-linearity.

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))  # Most of the time we use 2 by 2 for our pooling size

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu')) # in here we don't need input_shape parameter
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu')) # In here for units we can't use average of input output like before because in here our input is too big. by practice we realize that a number around 100 is good but since its better to choose a number in power of 2 then 128 it is.
classifier.add(Dense(units=1, activation='sigmoid')) # if our outcome was more than two, we use softmax function

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']) # There is two reason for using binary_crossentropy for our loss: 1. this function correspond to the logarithmic loss that is loss function we use in general for classification problems (like logistic regression) 2. having a binary outcome (otherwise categorical_crossentropy)

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator  # for having a good result we need lots and lots of data (in here we use 10'000 which is not a lot) or we can use a simple trick which is in data augmantation. it will create lots of batches of our images and in each batch it applies some radnom transformation on a random selection of images (like rotating, shifting, flipping or even shearing them). eventually we get more diverse images inside these batches and lot more material to train. # Image augmentation is a technique that allow us to enrich our dataset without adding more images and it allow us to get a good result with little or no overfitting (because of small number of pictures) even with a small amount of images. # copy past the code below from documantion of keras (preprocessing section in part of images) 



train_datagen = ImageDataGenerator(   # to prepare the image augmentation with creating a object
        rescale=1./255, # pixels take values between 0 to 255 in here after rescaling them our pixel values will be between 0 and 1 
        shear_range=0.2, # this applies random transaction 
        zoom_range=0.2, # applies some random zoom
        horizontal_flip=True) # it means that our image will flip horizontally

test_datagen = ImageDataGenerator(rescale=1./255)  # doing same as train_datagen but on our test set 




training_set = train_datagen.flow_from_directory('dataset/training_set',  # applying image aumentation on training set # this is from where we extract the images from
                                                 target_size=(64, 64),  # size of our images that is expected in CNN model. we give the same value as above
                                                 batch_size=32,  # size of our batch which some random samples of our images will be included and that contains the number of images that go through CNN after which the weights will be updated
                                                 class_mode='binary')
                                                    

test_set = test_datagen.flow_from_directory(                        # applying image aumentation on test set
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')



classifier.fit_generator(                  # fitting the training set to the CNN which also testing its performance on the test set 
                    training_set,
                    steps_per_epoch=8000, # number of images in training set
                    epochs=25, # number of epochs to train the CNN 
                    validation_data = test_set, # the test set that we want to evaluate the performance of CNN
                    validation_steps=2000)  # number of images in test set
# we optain an accuracy of 84% for training set and 75% on test set. the result are not too bad and not to good neither. 75% means that we get 3 correct answer out of 4. we can imporve this by having an accuracy of more than 80% for both and also getting a small diffrence between these two numbers. # to improve our model we should make a deeper deep learning model (deeper covolutional network) which for doing so we have two options 1. Add another convolutional layer 2. add another fully connected layer. so in here we will add second convolutional layer to our first one. remmber for doing so, add it right after Pooling
# after havin the second covolutional layer, we optain an accuracy of 85% for training set and 82% on test set. remmber for choosing between those 2 options for imporving our model, we can only realize which one is better by exprimenting. and also for imporving more we can choose higher target_size for our images. it is because we'are going to have a higher pixels.

"""
# Saving the model
classifier.save("classifier.h5")


# Loading the model
from keras.models import load_model
classifier = load_model('classifier.h5')

classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


"""

