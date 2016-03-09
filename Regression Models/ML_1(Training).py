
# coding: utf-8

# In[12]:
#Linear Regression with Plotting of Graphs
import numpy as np
import math
import csv

import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy.random import randn
from scipy import array, newaxis
input_file = csv.DictReader(open("ml_P1.csv"))

y = []
a = []
b = []
for row in input_file:
    a.append(float(row["A1"]))
    b.append(float(row["A2"]))
    y.append(float(row["P1"]))
    
flag  = 0
l = 0

post = 1000000000000
hj =0
n = input("Enter order: ")
r = input("Enter alpha: ")

if n == 1:
    t = [1,1,1]
    while (flag != 1):

        l = l +1
        d = []
        for i in range (len(a)):
            a1 = np.array([1,a[i],b[i]])
            b1 = np.array([t[0],t[1],t[2]])
            h = np.dot(a1,b1)
            d.append(h)
    
        nb = []
        f = []    
        m = float(len(y))
        #r = 0.03
        cost = 0.0
        um = float(2*m)
        con = float(1/um)

        cf = r/m
    
        for i in range (len(y)):
            d1 = d[i] - float(y[i])
            f.append(d1)
            p = d1*d1
            cost = cost + p
        
        cost = float(cost)*con
    #print(cost)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i]*1
            sm = sm + r1

        t1  = float(t[0]) - float(sm*cf)
        nb.append(t1)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i] * a[i]
            sm = sm + r1
    
        t2  = float(t[1]) - float(sm*cf)
        nb.append(t2)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * b[i])
            sm = sm + r1
    
        t3  = float(t[2]) - float(sm*cf)
        nb.append(t3)
    
   
        if post < cost:
            flag = 1
            print(post)
        else:
            post = cost
            t = nb

if n == 2:
    t = [1,1,1,1]
    while (flag != 1):

        l = l +1
        d = []
        for i in range (len(a)):
            a1 = np.array([1,a[i]*a[i],b[i]*b[i],a[i]*b[i]])
            b1 = np.array([t[0],t[1],t[2],t[3]])
            h = np.dot(a1,b1)
            d.append(h)
    
        nb = []
        f = []    
        m = float(len(y))
        #r = 0.03
        cost = 0.0
        um = float(2*m)
        con = float(1/um)

        cf = r/m
    
        for i in range (len(y)):
            d1 = d[i] - float(y[i])
            f.append(d1)
            p = d1*d1
            cost = cost + p
        
        cost = float(cost)*con
    #print(cost)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i]*1
            sm = sm + r1

        t1  = float(t[0]) - float(sm*cf)
        nb.append(t1)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i] * (a[i]*a[i])
            sm = sm + r1
    
        t2  = float(t[1]) - float(sm*cf)
        nb.append(t2)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (b[i]*b[i]))
            sm = sm + r1
    
        t3  = float(t[2]) - float(sm*cf)
        nb.append(t3)
    
   
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]))
            sm = sm + r1
    
        t4  = float(t[2]) - float(sm*cf)
        nb.append(t4)
    
        if post < cost:
            flag = 1
            print(post)
        else:
            post = cost
            t = nb

    
        
        
if n == 3:
    t = [1,1,1,1,1]
    while (flag != 1):

        l = l +1
        d = []
        for i in range (len(a)):
            a1 = np.array([1,a[i]*a[i]*a[i],b[i]*b[i]*b[i],a[i]*a[i]*b[i],b[i]*b[i]*a[i]])
            b1 = np.array([t[0],t[1],t[2],t[3],t[4]])
            h = np.dot(a1,b1)
            d.append(h)
    
        nb = []
        f = []    
        m = float(len(y))
        #r = 0.03
        cost = 0.0
        um = float(2*m)
        con = float(1/um)

        cf = r/m
    
        for i in range (len(y)):
            d1 = d[i] - float(y[i])
            f.append(d1)
            p = d1*d1
            cost = cost + p
        
        cost = float(cost)*con
    #print(cost)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i]*1
            sm = sm + r1

        t1  = float(t[0]) - float(sm*cf)
        nb.append(t1)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i] * (a[i]*a[i]*a[i])
            sm = sm + r1
    
        t2  = float(t[1]) - float(sm*cf)
        nb.append(t2)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (b[i]*b[i]*b[i]))
            sm = sm + r1
    
        t3  = float(t[2]) - float(sm*cf)
        nb.append(t3)
    
   
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]*a[i]))
            sm = sm + r1
    
        t4  = float(t[3]) - float(sm*cf)
        nb.append(t4)
        
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]*b[i]))
            sm = sm + r1
    
        t5  = float(t[4]) - float(sm*cf)
        nb.append(t5)
    
        if post < cost:
            flag = 1
            print(post)
        else:
            post = cost
            t = nb

if n == 4:
    t = [1,1,1,1,1,1]
    while (flag != 1):

        l = l +1
        d = []
        for i in range (len(a)):
            a1 = np.array([1,a[i]*a[i]*a[i]*a[i],b[i]*b[i]*b[i]*b[i],a[i]*a[i]*b[i]*b[i],b[i]*b[i]*a[i]*b[i],a[i]*a[i]*a[i]*b[i]])
            b1 = np.array([t[0],t[1],t[2],t[3],t[4],t[5]])
            h = np.dot(a1,b1)
            d.append(h)
    
        nb = []
        f = []    
        m = float(len(y))
        #r = 0.03
        cost = 0.0
        um = float(2*m)
        con = float(1/um)

        cf = r/m
    
        for i in range (len(y)):
            d1 = d[i] - float(y[i])
            f.append(d1)
            p = d1*d1
            cost = cost + p
        
        cost = float(cost)*con
    #print(cost)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i]*1
            sm = sm + r1

        t1  = float(t[0]) - float(sm*cf)
        nb.append(t1)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = f[i] * (a[i]*a[i]*a[i]*a[i])
            sm = sm + r1
    
        t2  = float(t[1]) - float(sm*cf)
        nb.append(t2)
    
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (b[i]*b[i]*b[i]*b[i]))
            sm = sm + r1
    
        t3  = float(t[2]) - float(sm*cf)
        nb.append(t3)
    
   
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]*a[i]*b[i]))
            sm = sm + r1
    
        t4  = float(t[3]) - float(sm*cf)
        nb.append(t4)
        
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]*b[i]*b[i]))
            sm = sm + r1
    
        t5  = float(t[4]) - float(sm*cf)
        nb.append(t5)
        
        sm = 0.0
        for i in range (len(y)):
            r1 = (f[i] * (a[i]*b[i]*a[i]*a[i]))
            sm = sm + r1
    
        t6  = float(t[5]) - float(sm*cf)
        nb.append(t6)
    
        if post < cost:
            flag = 1
            print(post)
        else:
            post = cost
            t = nb

#df = [18.0757,53.221,4.855,45.5481,30.157,46.437,41.418,24.897,30.757,11.053]
Xs = a
Ys = b
Zs = d


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show() # or:


# In[ ]:



