#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:12:15 2018

@author: chengch
"""

import numpy as np
import matplotlib.pyplot  as plt
import sklearn

from sklearn.datasets import load_iris

oringal_data = load_iris()
data = oringal_data['data']

def get_cov(data,k=1):
    for i in range(len(data[1])):
        data[:,i] = data[:,i] - np.mean(data[:,i])
    covariance_matrix = []
    for j in range(len(data[1])):
        for k in range(len(data[1])):
            dat = np.dot(data[:,j],data[:,k]) / (len(data) - 1)
            covariance_matrix.append(dat)
    covariance_matrix = np.reshape(covariance_matrix,(len(data[1]),len(data[1])))

    us,vs = np.linalg.eig(covariance_matrix)
    us_ind = np.argsort(-us)
    result = []
    for ind in range(k):
        result.append(vs[:,us_ind[ind]])
    result = np.array(result).T
    final_matrix = np.dot(data,result)
    return final_matrix

#test_data
t_d = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1],[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]]).T



#机器学习库中直接调用PCA
from sklearn.decomposition import PCA
pca = PCA(n_componets=2)
re = pca.fit_transform(t_d)
