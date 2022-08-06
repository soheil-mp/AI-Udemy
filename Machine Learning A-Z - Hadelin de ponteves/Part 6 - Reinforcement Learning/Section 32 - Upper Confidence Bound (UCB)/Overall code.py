
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importung the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv') # In this dataset you can see that, there is 10 diffrent version of ads shown for each person. 1 mean the person click on that ad. rememmber that this is a simulated data and in real life we don't have something like this. that is why reinforcement learning is also called online learning or interactive learning.

""" 
Random selection is the bad version of UCB because total reward is lower. it means that if we had a slot machine, we would score lower.
# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
"""

# Implementing UCB 
# there is no package for this. so, we should implement the whole agorithms by ourself

import math
N = 10000 # Total number of rounds. look at the dataset
d = 10 # Number of ads. look at the dataset
ads_selected = []
number_of_selections = [0] * d # this creates vector of size d. we initialize it ot zero because the first round of each version of the ad is zero # Step 1 part 1
sums_of_rewards = [0] * d # Step 1 part 2
total_reward = 0
for n in range(0, N): 
    ad = 0
    max_upper_bound = 0
    for i in range(0,d): # The reason we use if statement in here is because we want to deal with the initial condition that is what happens at round zero. the stretegy happens at round n and not at the first round (first 10 ads) we will use the strategy when we have our first 10 ads
        if (number_of_selections[i]>0):
            average_reward = sums_of_rewards[i] / number_of_selections[i]  # Step 2 Part 1
            delta_i = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i]) # Step 2 Part 2 # log(N+1) because index in python starts at 0 but log starts at 1 so we should go 1 ahead # rememmber that in here we only calculate the upper confidence bound and not the whole interval
            upper_bound = average_reward + delta_i # Step 2 Part 2 # Step 2 is just the upper bound for each of the diversion of the add at round n. that's why we create another variable that takes the maximum of these upper bounds of 10 ads here at round n.
        else:   # For the first round (10 ads), it goes to 'else'
            upper_bound = 1e400 # 10^400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i # To keep track of the index. remmber we always need to initialize so we put ad = 0 at start
            
    ads_selected.append(ad)  # After the first round we put the index inside the ads_selected
    number_of_selections[ad] = number_of_selections[ad] + 1 # because for each round we want to update our number_of_selections
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward  
    total_reward = total_reward + reward
    # Now if you run all of them and click on the ads_selected, you will see that for the first round (10 ads), the selected ad is itself. afterward strategy starts. if you see the last rounds, the number of ads are usually the same.
        
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlable('Ads')
plt.title('Number of times each ad was selected')
plt.show()  # The fifth ad is our best choice