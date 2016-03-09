
# coding: utf-8

# In[11]:

import csv
import math
import numpy as np
import collections
import heapq
from collections import Counter
import statistics
from statistics import mean


meany = []
meann =[]
vary = []
varn = []

print("\nTraining Files \t Test files \t Ratio")
print("\n train_spam \t spam_test \t 95-5 \n Train(50) \t Test(50) \t 50-50\n train2 \t Test2 \t\t 40-60")
nm = raw_input('\nEnter Training DataSet Name: ') 
name = nm + '.txt'

h = 57
for i in range(h):
    dy = []
    dn = []    
    with open(name, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row[h] == '1':
                dy.append(float(row[i]))
            else:
                dn.append(float(row[i]))
        
    meany.append(np.mean(dy))
    meann.append(np.mean(dn))
    vary.append(np.var(dy))
    varn.append(np.var(dn))
        

        
prio = float(len(dy))/float(len(dy)+len(dn))
print("Prior Probability of being spam : %s" % prio)
goal =[]
check = []


nm = raw_input('Enter Test DataSet Name: ') 
name_test = nm + '.txt'

with open(name_test, 'rb') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        ans1 = 1.0
        ans2 = 1.0
        check.append(float(row[57]))
        for i in range(h-1):
            a = float(row[i])
        
            avy = vary[i]
            amy = meany[i]
        
            v1 = math.pow(avy,2)
            va = 2*v1
            d1 = math.pow((a-amy),2)
            if d1 == 0 or v1 == 0:
                ex1 = 0
            else:
                ex1 = -(d1/v1)
            e1 = math.pow(math.e,ex1)
            if v1 == 0:
                sdn1 = 0
                pa = 0
            else:
                dn1 = 2*(math.pi)*v1
                sdn1 = math.sqrt(dn1)
                pa = float(e1)/float(sdn1)

        
            avn = varn[i]
            amn = meann[i]
            v2 = math.pow(avn,2)
            vb = 2*v2
            d2 = math.pow((a-amn),2)
            if v2 == 0:
                ex2 = 0
            else:
                ex2 = -(d2/v2)
            
            e2 = math.pow(math.e,ex2)
            dn2 = 2*(math.pi)*v2
            sdn2 = math.sqrt(dn2)
            if sdn2 == 0:
                pb = 0
            else:
                pb = float(e2)/float(sdn2)
        
            ans2 = pb * ans2 *(1-prio)
            ans1 = pa * ans1 *prio
        
        if ans1 > ans2:
            #print("Ans1 : %s" %(ans1))
            #print("Ans2 : %s" %(ans2))
            goal.append(1.0)
        else:
            #print("Ans1 : %s" %(ans1))
            #print("Ans2 : %s" %(ans2))
            goal.append(0.0)
        
        
        
        
        
        
cnt = 0        
for i in range(len(check)):
    if check[i] == goal[i]:
        cnt = cnt + 1
        

acc = (float(cnt)/len(check))*100
print("\nAccuracy: %s" % acc)
print("\nClass Count")
print(Counter(goal))
print(Counter(check))



#for name, count in Counter(goal).most_common(1):
#            c4,v4 = name,count

        
#for name, count in Counter(check).most_common(1):
#            c5,v5 = name,count
        

#TP = 
#FP = 
#TN = 
#FN =
        
        
#Precision = TP/(TP+FP)
#Recall = TP/(TP+FN)

#print("\nPrecision: %s" %(Precision))
#print("Recall: %s" % (Recall))



# In[ ]:



