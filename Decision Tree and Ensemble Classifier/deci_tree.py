
# coding: utf-8

# In[22]:

import numpy as np
import csv
import math
import collections
import heapq
from collections import Counter

#taking the values in matrix using numpy library
h = np.recfromcsv('mu_train.csv',delimiter=',')

#number of cloumns (last column is avoided as it directly predicts classes)
nc = 5


#different list for elements at different levels
#e = []
#initially flag  is 0 as it marks the levels
flag = 0


lev = []
lev1 = []
d =[]
di = []
y = []
#option1 = []
#y2 =[]
levv4 = []
levr4 = []
levr =[]
lev2 =[]
levn2 = []
lev3 =[]
levn3 =[]
lev4 = []
levn4 = []
lev5 = []
levn5 = []
levr5= []
levp5 = []
levc5 = []
#column 0 is the class
if flag == 0:
    opt = []
    for i in range(nc):
        e =[]
        if i == 0 :
            p = []
            for j in range(len(h)):
                p.append(h[j][i])
            
                
        else:
            a = []
            for j in range(len(h)):
                a.append(h[j][i])
            
            s = Counter(a)
            
            list1 = []
            s1 = []
            for name,cont in Counter(a).most_common(len(s)):
                g,b = name,cont
                list1.append(g)
                s1.append(b)
            
            
            for k in range (len(list1)):
                var1 = list1[k]
                ce =0
                cp =0
                for m in range (len(h)):
                    if h[m][i] == var1:
                        if h[m][0] == 'e':
                            ce = ce + 1
                        else:
                            cp = cp + 1
             
            
        
            
#calculating entropy for each unique value                        
                r1 = float(ce)/s1[k]
        
                if r1 == 0:
                    t = 0
                else:
                    t = -(math.log(r1,2))
            
            
                ent1 = r1 * t 
        
            
                r2 = float(cp)/s1[k]
                if r2 == 0:
                    t = 0
                else:
                    t = -(math.log(r2,2))
            

                ent2 = r2 * t
        
            
                ent = (ent1 + ent2)
            
                e.append(ent)
        
#calculating weighted entropy for each unique value and then adding it togeteher        
            sm = 0
            for o in range(len(e)):
                sm = sm + e[o]*(float(s1[o])/len(a))
      
            y.append(sm)

    for z in range (len(y)):
        print("Entropy of Column %s : %s" %((z+1),y[z]))
    #selecting feature that has minimum entropy
    feat1 = y.index(min(y)) + 1     #+1 is because 0th column we arent considering
    print("Column with lowest entropy: %s" % feat1)
    print("------level 0 complete------")
    lev1.append(feat1)
    flag = 1
        

print("\n\n")        
#calculating nodes for level 1
        
if flag == 1:
    a =[]
    
#taking all the values of first feature that we selected as root
    for j in range(len(h)):
        a.append(h[j][feat1])
                    
#getting all unique values and count in it
    s = Counter(a)
                
#getting values and corresponding count of each unique value in a column
    list1 = []
    s1 = []
    for name,cont in Counter(a).most_common(len(s)):
        g,b = name,cont
        list1.append(g)
        s1.append(b)
    
#taking each unique value and checking if it results in a class
#if not then we will get the next feature for that column

    for k in range (len(list1)):
        var1 = list1[k]
        y1 = []
        opt = []
        ce =0
        cp =0
        for m in range (len(h)):
            if h[m][feat1] == var1:
                if h[m][0] == 'e':
                    ce = ce + 1
                else:
                    cp = cp + 1
        
        #if it results in class that is either ce (count of edible) is 0 
        #or cp (count of poisonous) is 0 then we can get pure class or we further calculate entopy for next level
        
        if ce == 0:
            lev2.append("p")
            levn2.append(var1)
            print("For feature %s in column %s - Class: p" % ((var1),(feat1)))
            print("------level 1 complete------")
            print("\n")
            
        elif cp == 0:
            lev2.append("e")
            levn2.append(var1)
            print("For feature %s in column %s - Class: e" % ((var1),(feat1)))
            print("------level 1 complete------")
            print("\n")
        else:
            e1 = []
            #selecting the next node for the values whose class is not pure
            
            for i in range(nc):
                #if next column is already made a node earlier
                #we will not consider it
                if i in lev1:
                    tgddfdgd = 90
           
                #if the column is class value column(column 0)
                # then ignore it 
                elif i == 0:
                    ghdfdf = 98
                    
                    
                else:
                    c = []
                    #the column(feature) has not been made a node yet
                    for j in range(len(h)):
                        if h[j][feat1] == var1:
                            c.append(h[j][i])
                            
                    s = Counter(c)
                    x = len(c)
                    
                    list2 = []
                    s2 = []
                    #getting the unique values and counts each column(feature)
                    for name,cont in Counter(c).most_common(len(s)):
                        g,b = name,cont
                        list2.append(g)
                        s2.append(b)
          
                    #for every unique value along with root node getting the count of class 
                    for o in range(len(list2)):
                        ab = list2[o]
                        ce = 0
                        cp = 0
                        for j in range(len(h)):
                            #checking for var1 that is value of root node and ab that is current value together
                            #having occurence of 'e' or 'p'
                            if h[j][feat1] == var1:
                                if h[j][i] == ab:
                                    if h[j][0] == 'e':
                                        ce = ce + 1
                                    else:
                                        cp = cp + 1                        
                        
                        
                        #applying formula of entropy
                        r1 = float(ce)/s1[k]
                        
                        if r1 == 0:
                            t = 0
                        else:
                            t = -(math.log(r1,2))
            
            
                        ent1 = r1 * t 
        
            
                        r2 = float(cp)/s1[k]
                        if r2 == 0:
                            t = 0
                        else:
                            t = -(math.log(r2,2))
            
                        ent2 = r2 * t
        
            
                        ent = (ent1 + ent2)
                        
                        #taking all entropys(for each class) in a list
                        e1.append(ent)
                        
            
                    
                    sm = 0
                    for jo in range(len(s2)):
                        sm = sm + e1[jo]*(float(s2[jo])/x)
                        
                    #taking all entropys in a list for every column in a list    
                    y1.append(sm)
                    opt.append(i)

                    
                    
            for z in range(len(y1)):
                print("Entropy of Column %s : %s for %s" %(opt[z],y1[z],var1))
                
                    
                    
            #selecting next feature with the minimum value of entropy       
            feat3 = opt[y1.index(min(y1))]
            #adding it in general list that is used to check if we have made a feature or not
            lev1.append(feat3)
            lev3.append(opt[y1.index(min(y1))])
            levn3.append(var1)
            print("Column with lowest entropy: %s for %s" % (feat3,var1))
            print("------level 1 complete------")
            print("\n")
            flag = 2

            

            
#doing same thing for level 3
if flag == 2:
    for m in range(len(levn3)):
        var1 = levn3[m]
        indx = lev3[m]
        a =[]
        op =[]
        for j in range(len(h)):
            if h[j][feat1] == var1:
                a.append(h[j][indx])
        
        
        x = len(a)
        s = Counter(a)
        list3 = []
        s3 = []
        
        for name,cont in Counter(a).most_common(len(s)):
            g,b = name,cont
            list3.append(g)
            s3.append(b)
        
        for l in range (len(s3)):
            var2 = list3[l]
            e4 =[]
            opt = []
            ce = 0
            cp = 0
            for o in range(len(h)):
                if h[o][feat1] == var1:
                    if h[o][indx] == var2:
                        if h[o][0] == 'e':
                            ce = ce + 1
                        else:
                            cp = cp +1
                        
            if ce == 0:
                lev.append('p')
                #lev1.append(i)
                lev4.append(i)
                levv4.append('p')
                levn4.append(var2)
                levr4.append(var1)
                # attribute value [column number]
                print("Class for %s [%s] %s [%s] is 'p' " %(var1,feat1,var2,indx))
                print("------level 2 complete------")
                print("\n")

                
            elif cp == 0:
                lev.append('e')
                lev4.append(i)
                levv4.append('e')
                levn4.append(var2)
                levr4.append(var1)
                print("Class for %s [%s] %s [%s] is 'e' " %(var1,feat1,var2,indx))
                print("------level 2 complete------")
                print("\n")
                
        
            else:

                #print("%s %s %s %s %s" % (var1,var2,ce,cp,s3[l]))
                for i in range(nc):
                    if  i == 0:
                        etr = 90
                    elif i in lev1:
                        etc = 9
                    else:
                        p = []
                        for j in range (len(h)):
                            if h[j][feat1] == var1:
                                if h[j][indx] == var2:
                                    p.append(h[j][i])
                            
                        s = Counter(p)
                        s4 = []
                        list4 = []
                        
                        #print(s)
                        x = len(p)
                        for name,cont in Counter(p).most_common(len(s)):
                            g,b = name,cont
                            list4.append(g)
                            s4.append(b)
                            
                        
                        for k in range(len(list4)):
                            var5 = list4[k]
                            ce = 0
                            cp = 0
                            for j in range(len(h)):
                                if h[j][feat1] == var1:
                                    if h[j][indx] == var2:
                                        if h[j][i] == var5:
                                            if h[j][0] == 'e':
                                                ce = ce +1
                                            else:
                                                cp = cp +1
                        
                            
                            #check if only one column is left to be selected
                            if len(lev1) == (nc-2):                             
                                if ((ce == 0) or (cp> ce)):
                                    lev5.append(i)
                                    levp5.append(var1)
                                    levr5.append(var2)
                                    levc5.append(var5)
                                    levn5.append('p')
                                    print("Class for %s [%s] %s [%s] %s [%s] is 'p' " %(var1,feat1,var2,indx,var5,i))
                                    print("------level 3 complete------")
                                    
                                elif  ((cp == 0) or (ce > cp)) :
                                    lev5.append(i)
                                    levp5.append(var1)                 
                                    levr5.append(var2)
                                    levc5.append(var5)
                                    levn5.append('e')
                                    print("Class for %s [%s] %s [%s] %s [%s] is 'e' " %(var1,feat1,var2,indx,var5,i))
                                    print("------level 3 complete------")
                                
                                else:
                                    lev5.append(i)
                                    levp5.append(var1)
                                    levr5.append(var2)
                                    levc5.append(var5)
                                    levn5.append('e')
                                    print("Class for %s [%s] %s [%s] %s [%s] is 'e' " %(var1,feat1,var2,indx,var5,i))
                                    print("------level 3 complete------")
                            else:
                                flag = 10009

                                
                                
                                
print("\n\n")
                            
#Testing on Training Data
#-------------------------------------------------------------
a = []
h = np.recfromcsv('mu_train.csv',delimiter=',')
for i in range (len(h)):
    if h[i][feat1] in levn2:
        a.append(lev2[0])
    else:
        for j in range(len(lev3)):
            ind = lev3[j]
            if h[i][feat1] == levn3[j]:
                if h[i][ind] in levn4 :
                    a.append(levv4[0]) 

                else:
                    for k in range(len(levr5)):
                        if h[i][ind] == levr5[k]:
                            if h[i][lev5[k]] == levc5[k]:
                                a.append(levn5[k])
                                
            
            
            

c =0
for j in range(len(a)):
    if a[j] == h[j][0]:
        c = c +1
    

acc = (float(c)/len(h)) * 100
print("Accuracy on train data: %s" % (acc))
                            
                            
            


#Testing on Test Data
#-------------------------------------------------
a = []
h = np.recfromcsv('mu_test.csv',delimiter=',')

#check it for eac and every element
for i in range (len(h)):
    if h[i][feat1] in levn2:
        a.append(lev2[0])
    else:
        for j in range(len(lev3)):
            ind = lev3[j]
            if h[i][feat1] == levn3[j]:
                if h[i][ind] in levn4 :
                    a.append(levv4[0]) 

                else:
                    for k in range(len(levr5)):
                        if h[i][ind] == levr5[k]:
                            if h[i][lev5[k]] == levc5[k]:
                                a.append(levn5[k])
                                
            
            
            

c =0
for j in range(len(a)):
    if a[j] == h[j][0]:
        c = c +1
    

acc = (float(c)/len(h)) * 100
print("Accuracy on test data: %s" % (acc))



# In[ ]:



