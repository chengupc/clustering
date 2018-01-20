import numpy as np
from PIL import Image
from numpy import *

img = Image.open('/home/chengch/test.png')
img = img.resize((200,200))
img.show()
img_array = np.array(img)

def pca(data_mat, top_n_feat=99999999):

    # 获取数据条数和每条的维数 
    num_data,dim = data_mat.shape  
    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  #shape:(784,)
    mean_removed = data_mat - mean_vals # shape:(100, 784)

    # 计算协方差矩阵（Find covariance matrix）
    cov_mat = np.cov(mean_removed, rowvar=0) # shape：(784, 784)

    # 计算特征值(Find eigenvalues and eigenvectors)
    eig_vals, eig_vects = np.linalg.eig(mat(cov_mat)) # 计算特征值和特征向量，shape分别为（784，）和(784, 784)

    eig_val_index = argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1) : -1] # 最大的前top_n_feat个特征的索引
    # 取前top_n_feat个特征后重构的特征向量矩阵reorganize eig vects, 
    # shape为(784, top_n_feat)，top_n_feat最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index] 
    
    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects # shape: (100, top_n_feat), top_n_feat最大为特征总数
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals # 根据前几个特征向量重构回去的矩阵，shape:(100, 784)
    
    return low_d_data_mat, recon_mat

low_d_feat_for_7_imgs, recon_mat_for_7_imgs = pca(img_array)
re_img = Image.fromarray(recon_mat_for_7_imgs)
re_img.show()