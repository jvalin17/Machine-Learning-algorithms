
# coding: utf-8

# In[22]:

import numpy as np
import math
import csv



#name is name of test file that is Du in the homework
name = 'test_ml3.csv'
#t is theta values ; randomly taken 
t = [1,2,3,4]

loop = 0
#k denotes the number of times we run self training algorithm to classify unlabled data
k = 10
for loop in range(k):
    mn1 = post
    loop = loop +1 
    #input file is labeled file Ds stored as 'MLp.csv'
    input_file = csv.DictReader(open("ML_p3.csv"))

    ht = []
    w = []
    a = []
    s = []
    #reads data from csv file to different lists
    for row in input_file:
        ht.append(float(row["H"]))
        w.append(float(row["W"]))
        a.append(float(row["A"]))
        s.append(float(row["S"]))
        
    
    
    l = 0
    #applying logistic regression for 500 times to get stable value of theta
    while (l != 500):
        l = l +1
        d = []

        for i in range (len(w)):
            #reads values from lists to a new list
            a1 = np.array([1,ht[i],w[i],a[i]])
            #gets value of theta from t which gets updated after every loop
            b1 = np.array([t[0],t[1],t[2],t[3]])
            j9 = b1.transpose()
            #calculating h(theta) 
            h = np.dot(a1,b1.T)
            #applying logistic regression formula
            fh = math.e ** (-1.0 * h)
            fh = 1.0 + fh
            lgst = 1.0/fh
        #Storing all the logistic regression values for each row of Ds
            d.append(lgst)
    
        nb = []
        f = []
        #m is number of records
        m = float(len(s))
        # r is the learning rate 
        r = 0.03
        #initial cost is 0
        cost = 0.0
        um = float(2*m)
        con = float(1/um)

        cf = r/m
        #Calculating Cost
        #s is a list that contains values of actual class (Male= 1 Female = 0)
        for i in range (len(s)):
            d1 = d[i] - float(s[i])
            f.append(d1)
            p = d1*d1
            cost = cost + p
        
        cost = float(cost)*con
        #applying gradient descent to each of the attribute
        
        sm = 0.0
        for i in range (len(s)):
            r1 = f[i]*1
            sm = sm + r1

        t1  = float(t[0]) - float(sm*cf)
        nb.append(t1)
    
        sm = 0.0
        for i in range (len(s)):
            r1 = f[i] * ht[i] #ht has height values
            sm = sm + r1
    
        t2  = float(t[1]) - float(sm*cf)
        nb.append(t2)
    
        sm = 0.0
        for i in range (len(s)):
            r1 = (f[i] * w[i]) #w has weight values
            sm = sm + r1
    
        t3  = float(t[2]) - float(sm*cf)
        nb.append(t3)
    
        sm = 0.0
        for i in range (len(s)):
            r1 = (f[i] * a[i]) #a has age values
            sm = sm + r1
    
        t4  = float(t[3]) - float(sm*cf)
        nb.append(t4)
    
        #t1,t2,t3,t4 are values of t i.e. theta1, theta2 and so on
        #are appended in a list nb which is then assigned to t
        #It is repeated for 500 times here
        t = nb
        post = cost
        

    #Printing cost after each iteration of self training algorithm
    print("Cost : %s" %post)
    #mn = post
    #Open a file which is named as 'name' which contains unlabeled data
    input_file = csv.DictReader(open(name))
    #we apply the value of theta learned from previous logistic regression algorithm to the unlabeled data 
    ht1 = []
    w1 = []
    ag1 = []
    st1 = []
    for row in input_file:
        ht1.append(float(row["H"]))
        w1.append(float(row["W"]))
        ag1.append(float(row["A"]))
    
    #new file is created with new name and name is updated
    name = 'at' + str(loop) + '.csv' #naming uniquely via loop
    with open(name, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["H","W","A"]) #headings of column is written as one row

    #calculation of h(theta) 
    for i in range (len(w1)):
        a1 = np.array([1,ht1[i],w1[i],ag1[i]])
        b1 = np.array([t[0],t[1],t[2],t[3]])
        j9 = b1.transpose()
        h = np.dot(a1,b1.T)
        fh = math.e ** (-1.0 * h)
        fh = 1.0 + fh
        lgst = 1.0/fh
        
        #threshold is 0.5 i.e if value of h(theta) is >= 0.5 then it is classified as 1 and updated in the file of labeled 
        #data. Else it is appended in a new file with new name that we created above and it is used in next loop
        if lgst >= 0.5:
            with open('ML_p3.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow([ht1[i],w1[i],ag1[i],lgst])
        else:
            with open(name, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow([ht1[i],w1[i],ag1[i]])

        #above is process is repeated for k times and value of theta that produces lowest cost is selected as final theta.

        
print("\nFinal Theta")
print(t)


# In[ ]:



