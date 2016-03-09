
# coding: utf-8

# In[17]:
#Logistic Regression
import numpy as np
import math
import csv



input_file = csv.DictReader(open("ml_P2.csv"))

ht = []
w = []
a = []
s = []
for row in input_file:
    ht.append(float(row["H"]))
    w.append(float(row["W"]))
    a.append(float(row["A"]))
    s.append(float(row["S"]))
    


post = 1000000000
t = [1,1,1,1]
l = 0
flag =0
while (flag != 1 ):
    l = l +1
    d = []
    for i in range (len(a)):
        a1 = np.array([1,ht[i],w[i],a[i]])
        b1 = np.array([t[0],t[1],t[2],t[3]])
        h = np.dot(a1,b1)
            
        fh = math.pow(math.e,-h)
        deno = float(1+fh)
        lgst = 1/deno
        d.append(lgst)
        
    nb = []
    f = []
    f1 = []
    f2 =[]
    f3 =[]
    m = float(len(s))
    r = 0.03
    cost = 0.0
    um = float(2*m)
    con = float(1/um)

    cf = r/m
    
    for i in range (len(s)):
        if s[i] == 1:
            d1 = d[i] - float(s[i])
            f.append(d1)
            f1.append(ht[i])
            f2.append(w[i])
            f3.append(a[i])
            p = d1*d1
            cost = cost + p
        
    cost = float(cost)*con
    
    
    sm = 0.0
    for i in range (len(f)):
        r1 = f[i]*1
        sm = sm + r1

    t1  = float(t[0]) - float(sm*cf)
    nb.append(t1)
    
    sm = 0.0
    for i in range (len(f)):
        r1 = f[i] * f1[i]
        sm = sm + r1
    
    t2  = float(t[1]) - float(sm*cf)
    nb.append(t2)
    
    sm = 0.0
    for i in range (len(f)):
        r1 = (f[i] * f2[i])
        sm = sm + r1
    
    t3  = float(t[2]) - float(sm*cf)
    nb.append(t3)
        
    sm = 0.0
    for i in range (len(f)):
        r1 = (f[i] * f3[i])
        sm = sm + r1
    
    t4  = float(t[3]) - float(sm*cf)
    nb.append(t4)
    
   
    if post < cost:
        flag = 1
    else:
        post = cost
        t = nb

        
print(t)

#Testing the Theta values
tst_h = [162,168,175,180]
tst_w = [53,75,70,85]
tst_a = [28,32,30,29]

for i in range(len(tst_h)):
    t1 = np.array([1,tst_h[i],tst_w[i],tst_a[i]])
    t2 = t
    h = np.dot(t1,t2)
       
    fh = long(math.pow(math.e,-h))
    
    deno = long(1+fh)
    lgst = 1/deno
    
    
    if lgst > 0.5:
        print("Prediction: M  h(x) = %s " % (lgst))
    else:
        print("Prediction: W  h(x) = %s" % (lgst))
    
    


# In[ ]:



