#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:20:44 2018

@author: chengch
"""
import numpy as np
import math
import matplotlib.pyplot as plt

data = np.loadtxt('/home/chengch/science/test.txt')
k=3
t_dat = data[20]

def normal(data):
    normal_dats = []
    min_dat = min(data)
    max_dat = max(data)
    for dat in data:
        normal_dats.append((dat - min_dat) / (max_dat - min_dat))
    normal_dats = np.array(normal_dats)
    normal_dats = np.reshape(normal_dats,(len(normal_dats),1))
    return normal_dats

def distance(x1,x2):
    dis = np.sqrt(sum((x1 - x2) ** 2))
    return dis

#density
dis_num = []
for i,dat in enumerate(data):
    n = 0
    for j in data:
        if distance(dat,j) < 1.5:
            n += 1
    dis_num.append((i,n))
    
dis_num = sorted(dis_num,key=lambda x:x[1],reverse=True)
dis_num = np.array(dis_num)

#distance
diss = []
for i,dat in enumerate(dis_num):
    if i == 0:
        first_dat_ind = dat[0]
        first_diss = []
        for j in data:
            first_diss.append(distance(data[dat[0]],j))
        first_diss = sorted(first_diss)
        dis = first_diss[-1]
        diss.append(dis)
    elif i == 1:
        second_dat_ind = dat[0]
        dis = distance(data[first_dat_ind],data[second_dat_ind])
        diss.append(dis)
    else:
        distance_all = []
        for k in dis_num[:,0][:i]:
            distance_all.append(distance(data[dis_num[i][0]],data[k]))
        dis = min(distance_all)
        diss.append(dis)
diss = np.array(diss)
diss = np.reshape(diss,(len(diss),1))
dis_den = np.concatenate((dis_num,diss),axis=1)

gama = normal(dis_den[:,1]) * normal(dis_den[:,2])
point_ind = np.reshape(dis_den[:,0],(len(dis_den[:,0]),1))
result = np.concatenate((point_ind,gama),axis=1)
result = np.array(sorted(result,key=lambda x:x[1],reverse=True))

k_values = result[:,0][:3]
cluster_centers = []
for i in k_values:
    cluster_centers.append(data[i])
cluster_centers = np.array(cluster_centers)

plt.scatter(data[:,0],data[:,1],color='b',s=50)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],color='r',s= 50)
plt.savefig('/home/chengch/science/science.png',dpi=300)
plt.show()



