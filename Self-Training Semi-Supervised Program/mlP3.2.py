
# coding: utf-8

# In[5]:

#1- Male and 0-Female
#Naive Bayes
import csv
import math
import numpy as np
import collections
import heapq
from collections import Counter
import statistics
from statistics import mean

a =[]
w =[]
h =[]
s =[]
r = []
rx = []

input_file = csv.DictReader(open("ml3_Ds.csv"))
#read values from csv file and store in list
for row in input_file:
    a.append(int(row["A"]))
    w.append(int(row["W"]))
    h.append(int(row["H"]))
    s.append(int(row["S"]))
    

f1 =[]
f2 =[]
f3 =[]
p1 =[]
p2 =[]
p3 =[]

c1 =0
c2 =0

for i in range (len(a)):
    #collects all records that has female
    if s[i] == 0 :
        f1.append(a[i])
        f2.append(w[i])
        f3.append(h[i])
        c1 = c1 + 1
    else:
        #collects records that has male
        c2 = c2 + 1
        p1.append(a[i])
        p2.append(w[i])
        p3.append(h[i])
        
#Calculates prior probability
prio1 = float(c1)/(len(a))
prio2 = float(c2)/(len(a))

#Calulates mean for each of the attribute with different class
mean_aw = np.mean(f1)
mean_ww = np.mean(f2)
mean_hw = np.mean(f3)
mean_am = np.mean(p1)
mean_wm = np.mean(p2)
mean_hm = np.mean(p3)

#Calulates variance for each of the attribute with different class
var_aw = np.var(f1)
var_ww = np.var(f2)
var_hw = np.var(f3)
var_am = np.var(p1)
var_wm = np.var(p2)
var_hm = np.var(p3)


ag = []
wg = []
hg = []
sg = []
#opens the test file that id Dt
input_file = csv.DictReader(open("ml3_2.csv"))

for row in input_file:
    ag.append(int(row["A"]))
    wg.append(int(row["W"]))
    hg.append(int(row["H"]))
    sg.append(int(row["S"]))

#calculates the the  naive bayes probability for each attribute
for i in range (len(ag)):
    v1 = math.pow(var_aw,2)
    v2 = math.pow(var_ww,2)
    v3 = math.pow(var_hw,2)
    va = 2*v1
    vw = 2*v2
    vh = 2*v3
    d1 = math.pow((ag[i]-mean_aw),2)
    d2 = math.pow((wg[i]-mean_ww),2)
    d3 = math.pow((hg[i]-mean_hm),2)
    ex1 = -(d1/v1)
    ex2 = -(d2/v2)
    ex3 = -(d3/v3)
    e1 = math.pow(math.e,ex1)
    e2 = math.pow(math.e,ex2)
    e3 = math.pow(math.e,ex3)
    dn1 = 2*(math.pi)*v1
    dn2 = 2*(math.pi)*v2
    dn3 = 2*(math.pi)*v3
    sdn1 = math.sqrt(dn1)
    sdn2 = math.sqrt(dn2)
    sdn3 = math.sqrt(dn3)
    pa = float(e1)/float(sdn1)
    pw = float(e2)/float(sdn2)
    ph = float(e3)/float(sdn3)
    prob_w = pa*pw*ph*prio1 #multiples each probability and prior probability

#Repeat same steps for other class
    v1 = math.pow(var_am,2)
    v2 = math.pow(var_wm,2)
    v3 = math.pow(var_hm,2)
    va = 2*v1
    vw = 2*v2
    vh = 2*v3
    d1 = math.pow((ag[i]-mean_am),2)
    d2 = math.pow((wg[i]-mean_wm),2)
    d3 = math.pow((hg[i]-mean_hm),2)
    ex1 = -(d1/v1)
    ex2 = -(d2/v2)
    ex3 = -(d3/v3)
    e1 = math.pow(math.e,ex1)
    e2 = math.pow(math.e,ex2)
    e3 = math.pow(math.e,ex3)
    dn1 = 2*(math.pi)*v1
    dn2 = 2*(math.pi)*v2
    dn3 = 2*(math.pi)*v3
    sdn1 = math.sqrt(dn1)
    sdn2 = math.sqrt(dn2)
    sdn3 = math.sqrt(dn3)
    pa1 = float(e1)/float(sdn1)
    pw1 = float(e2)/float(sdn2)
    ph1 = float(e3)/float(sdn3)
    prob_m = pa1*pw1*ph1*prio2

#Checks which has higher probabilty and classifies the rcord as that class
    if prob_w > prob_m:
        r.append(0)  
    else:
        r.append(1)
        
        
#Testing Result
#sg has the actual class of each record so it compares and gives result.
cnt = 0
for i in range(len(r)):
    if r[i] == sg[i]:
        cnt = cnt + 1
        

acc = float(cnt)/float(len(sg))
print("Accuracy with Naive Bayes Classifier: %s" % (acc * 100))  


#--------------------------------------------------------------------------------------------------------------------
#Semi Supervised Logistic Regression


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
    loop = loop +1 
    #input file is labeled file Ds stored as 'MLp.csv'
    input_file = csv.DictReader(open("MLp.csv"))

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
    #*** 
    name = 'check' + str(loop) + '.csv' #naming uniquely via loop
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
            with open('MLp.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow([ht1[i],w1[i],ag1[i],lgst])
        else:
            with open(name, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow([ht1[i],w1[i],ag1[i]])

        #above is process is repeated for k times and value of theta that produces lowest cost is selected as final theta.

#Testing of Semi-Supervised Algorithm on Test Data
#Testing on  Given Test Data
input_file = csv.DictReader(open("ml3_2.csv"))

ht = []
w = []
a = []
s = []
r = []
for row in input_file:
    ht.append(float(row["H"]))
    w.append(float(row["W"]))
    a.append(float(row["A"]))
    s.append(float(row["S"]))

#We apply the best values of theta and get h(theta) which clsssifies the record
for i in range (len(w)):
    a1 = np.array([1,ht[i],w[i],a[i]])
    b1 = np.array([t[0],t[1],t[2],t[3]])
    j9 = b1.transpose()
    h = np.dot(a1,b1.T)
    fh = math.e ** (-1.0 * h)
    fh = 1.0 + fh
    lgst = 1.0/fh
        
    #if value of h(theta) is greater than 0.5 then it is classified as male(1) 
    if lgst >= 0.5:
        r.append(1)
    else:
        r.append(0)
            
#calculating accuracy            
cnt = 0
for i in range(len(r)):
    if r[i] == s[i]:
        cnt = cnt + 1
        
acc = float(cnt)/float(len(sg))
print("Accuracy with Semi-supervised Learning: %s" % (acc * 100))       

