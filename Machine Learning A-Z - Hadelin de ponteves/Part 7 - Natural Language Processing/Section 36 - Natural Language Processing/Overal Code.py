
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t' , quoting=3)  # the reason we use '\t' is because we using a tab delimiter inside our file. tsv stands for tab seperated values. csv stands for comma seperated values. # quote = 3 means that we are ignoring double quotes because they sometimes make some problems

# Cleaning the texts
import re
import nltk
nltk.download('stopwords') # In here we are trying to eliminating all of the unrelated words like: the, that, and, in, etc. which are articles, propositions and so on. because it doesn't help our algorithm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # This is called stemming. in here we want only root of the word. for example we don't need loved, loving, etc. we only need love.
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # The sub method will eliminate every disirable parameters (for example dots, question mark,...). for doing so just simply add the parameter type inside it. the hat ^ means that eliminate everything except the parameter inside. in here we want to eliminate everything (even numbers) except letters from a-z and A-Z and SPACE (because we don't want to create a new word)
    review = review.lower() # To make all letters lowercase
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # The reason we are using set is because it makes our algorithms faster (than a list alone). the imporovement are mostly obvious in articles and books and not so much in reviews because it is short
    review = ' '.join(review) # For turning the list into a string
    corpus.append(review)
    
# Creating the bag of Words model  # We are creating it through process of Tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # max_feature will only keep the more used or relavent words but based on the number you give to it. this will also reduce the sparcity. in general there are two ways for reducing the sparcity: 1- max_features 2- dimensionality reduction. 
X = cv.fit_transform(corpus).toarray() # In here we are making a huge Sparse Matrix. # The reason we use toarray() is because we want to have a matrix
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(55+91)/200 # Acuracy