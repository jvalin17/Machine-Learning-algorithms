
# coding: utf-8

# In[7]:

#Naive Bayes on continuous attributes

import csv
import math
import numpy as np
import collections
import heapq
from collections import Counter


from random import randint
import random

#list declarations for getting mean and variance of each column

mean_spam = []
mean_non_spam =[]
var_spam = []
var_non_spam = []


#taking base file in matrix

fileload = np.loadtxt('spambase.txt')

#gettin number of columns of the file

cols = fileload[1]
num_cols = len(cols)

#getting split percentage from user

split_percentage = raw_input('\nEnter Split Percentage for Training Data: ') 
split_ratio = int(split_percentage)
len_rows = len(fileload)

#value of split percentage

prcnt_to_num = int((split_ratio * len_rows)/100)

#getting random unbiased unique rows for training data
#t = training_range
training_range = random.sample(range(0, len_rows), prcnt_to_num)


#making two files for train and test
#file_1 = open('train.txt', 'w+')
#file_2 = open('test.txt','w+')
    
#splitting rows in each of the file for training and testing
for i in range(len(fileload)):
    if i in t:
        with open('train.txt', 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t')
            spamwriter.writerow(hp[i])
    else:
        with open('test.txt', 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t')
            spamwriter.writerow(hp[i])
        
#in files the index start from 0 so subtracting 1 from total columns
cols_to_read = num_cols-1
resultant_col = num_cols-1

#for every column getting mean and variance based on the class value
for i in range(cols_to_read):
    spam = []
    non_spam = []    
    with open('train.txt', 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            #if its a spam 
            if row[resultant_col] == '1.0':
                spam.append(float(row[i]))
            else:
                #not a spam
                non_spam.append(float(row[i]))
    
    #using numpy for mean of each class and each column
    mean_spam.append(np.mean(spam))
    mean_non_spam.append(np.mean(non_spam))
    var_spam.append(np.var(spam))
    var_non_spam.append(np.var(non_spam))
        

#calculating prior probability based on training data

prio = float(len(spam))/float(len(spam)+len(non_spam))
print("Prior Probability of being spam : %s" % prio)

#goal[] to store the predicted values based on classifier
goal =[]
#check[] will have actual values
check = []



with open('test.txt', 'rb') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        #to multiply independent probabilities of every column 
        ans1 = 1.0
        ans2 = 1.0
        #getting actual class values
        check.append(float(row[h]))
        #not taking the last column as its class value
        #checking the probability for each row to be spam and to not to be spam
        for i in range(resultant_col-1):
            read_cell = float(row[i])
        
            read_var_spam = var_spam[i]   
            read_mean_spam = mean_spam[i]  
        
            square_var = math.pow(read_var_spam,2)
            twice_sq_var = 2*square_var
            deno = math.pow((read_cell - read_mean_spam),2)
            
            if deno == 0 or square_var == 0:
                exponent1 = 0
            else:
                exponent1 = -(deno/twice_sq_var)
                
            e1 = math.pow(math.e,exponent1)
            if read_cell == 0:
                sqrt_deno = 0
                prob_spam = 0
            else:
                deno_non_spam = 2*(math.pi)*v1
                sqrt_deno = math.sqrt(deno_non_spam)
                prob_spam = float(e1)/float(sqrt_deno)

        
            avar_non_spam = varn[i]
            amean_non_spam = meann[i]
            non_spam_val = math.pow(avar_non_spam,2)
            
            vb = 2*non_spam_val
            d2 = math.pow((read_cell-amean_non_spam),2)
            
            
            if non_spam_val == 0:
                exponent2 = 0
            else:
                exponent2 = -(d2/v2)
            
            e2 = math.pow(math.e,exponent2)
            deno_n = 2*(math.pi)*v2
            sqrt_deno2 = math.sqrt(deno_n)
            if sqrt_deno2 == 0:
                prob_non_spam = 0
            else:
                prob_non_spam = float(e2)/float(sqrt_deno2)
            #ans2 has probability of non-spam so multiplying with (1-prior_probability)
            ans2 = prob_non_spam * ans2 *(1-prio)
            #ans1 has probability of spam so multiplying with prior probability
            ans1 = prob_spam * ans1 *prio
        
        #checking higher probability and assigning corresponding value
        if ans1 > ans2:
            goal.append(1.0)
        else:
            goal.append(0.0)
        
        
        
        
        
        
#checking how many are correctly classified       
cnt = 0        
for i in range(len(check)):
    if (float(check[i]) == float(goal[i])):
        cnt = cnt + 1
        

acc = (float(cnt)/len(check))*100
print("\nAccuracy: %s" % acc)


print('\nTotal Rows: %s ' %(ln))
print('Number of rows for Training: %s' %(prc))

print("\nClass Count (0.0 - Non-Spam , 1.0 - Spam)")
print("\nClassified Values: ")
print(Counter(goal))
print("Actual Values: ")
print(Counter(check))







# In[ ]:



