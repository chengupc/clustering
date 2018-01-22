#test data
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50)

from sklearn.cluster import KMeans
est = KMeans(4)  # 4 clusters
est.fit(X)
y_kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')

#mnist dataset test
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import random

random.seed(2)

labels = np.load('/home/chengch/hand_write_data/labels.npy')[:2000]
images = np.load('/home/chengch/hand_write_data/images.npy')[:2000]

dat0 = [];dat1 = [];dat2 = []; dat3 = []

for i, dat in enumerate(labels):
    if dat == 0:
        dat0.append(images[i])
    elif dat == 1:
        dat1.append(images[i])
    elif dat == 2:
        dat2.append(images[i])
    elif dat == 3:
        dat3.append(images[i])
#       
dat0 = np.array(dat0);dat1 = np.array(dat1);dat2 = np.array(dat2);dat3 = np.array(dat3)
dat0 = dat0[:170];dat1 = dat1[:170];dat2 = dat2[:170];dat3 = dat3[:170]
total_dat = np.concatenate((dat0,dat1,dat2,dat3))
total_ind = np.concatenate((np.zeros((170,1)),np.ones((170,1)),np.ones((170,1)) * 2 , np.ones((170,1)) * 3))
total_data = np.concatenate((total_dat,total_ind),axis=1)
random.shuffle(total_data)
#
input_data = total_data[:, :-1]
target = total_data[:,-1]
y_pred = KMeans(n_clusters=4,random_state=0).fit_predict(input_data)


