
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None) # because the person who gather the data, falsely put the info on header.
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the dataset
from apyori import apriori # the library is our file inside the folder
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)   
# minimum support: in here we choose products that are purchesed 3 (or 4) times a day. the number we choose depends on our business data. 3*7/7500 = 0.0028 we consider the products that are purchased 3 times a day + total number of transactions over a week / total number of transactions 
# min_confidence: the reason we choose 20% is because if we use the default value of 80% (the rules are true for 80% of the times), we will get some false ilogical association for example in summer in France, alot of people buy mineral water with eggs and 80% confidence takes it as logic association between them, eventhough there is no such thing 
# min_length: we want minumum number of  two product in our bascket
# min_lift: minimum number of 3 seems the best here but again it depends on our dataset. you can find the best number by actually doing it

# Visualising the results
result = list(rules)

# if we double click on result we will see the rules that are sorted by their relevance
# The first row is the top relavent rule
# we can't see the rules on firs table. for seeing the rules click on the Value
# second table: 
# The first row is association between light cream and chicken that is people who bought light cream also bought chicken
# The second row is the support of this association rule which is the proportion of the set of products (cream and chicken) among other transactions
# click on the third row an go to third table. click again on the first row.
# Forth table:
# Left hand side of rule is cream. right hand side of rule is chicken. it means that if people buy cream, they are also likely buy chicken.
# third row is our confidence of rule which is 29%. it means that if people buy light cream, there is a 29% chance that they will buy chicken
# Forth row is our lift which is 4.8 and that is good because we are looking at lifts that are above 3 and this shows relavance of the rule (or in other word, it make sense).