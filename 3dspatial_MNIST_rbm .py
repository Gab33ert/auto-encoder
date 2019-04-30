# topographique auto encoder sturcture
import os
import tqdm
import voronoi
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import time

import topology as top
import function as func
import analyzeTools as at
import backprop
import rbm_train as rbmt
import rbm_visualize as rbmv





 

#global variable
size=784
n_cell=1600
dataset_size=1000
dataset_size_t=300
t=8
sigma=5#90
d=100#260
alpha=0.0005#backprop rate

epsilon=0.001#rbm rate


P, W,in_index =top.build_3d(60, d, sigma)
Wt=top.abstract_layer(in_index, W, t)
#for i in range(t-1,0,-1):
    #at.analyze_topology(Wt, i)
#for i in range(t-1,0,-1):
    #at.analyze_topology_froward(Wt,i)
#for i in range(t-1,0,-1):
#    at.analyze_topology_back(Wt,i)
    
"""
f, axarr = plt.subplots(4, sharex=True)                                        #histogram of the number of output neurone recieving 1,2... input neurone at each layer
axarr[0].hist(np.sum(Wt[0],axis=1), bins=np.max(np.sum(Wt[0],axis=1)))
axarr[1].hist(np.sum(Wt[1],axis=1), bins=np.max(np.sum(Wt[1],axis=1)))
axarr[2].hist(np.sum(Wt[2],axis=1), bins=np.max(np.sum(Wt[2],axis=1)))
axarr[3].hist(np.sum(Wt[3],axis=1), bins=np.max(np.sum(Wt[3],axis=1)))
plt.show()
"""
w=[]                                                                           #initialise weight and bias
b=[]#forward bias
c=[]#backward bias
for i in range(len(Wt)):
    wc=copy.deepcopy(Wt[i])
    #wc=np.ones(wc.shape)
    wc=wc*np.random.normal(0, 0.01, wc.shape)#(2*np.random.random(wc.shape)-1)*0.01
    w.append(wc)
    b.append(np.zeros((wc.shape[1],1)))
    c.append(np.zeros((wc.shape[0],1)))
b[0]=np.random.randn(Wt[0].shape[1],1)



#wc=np.load("test.npy")
#W=copy.deepcopy((wc != 0 ).astype(int))

mnist = tf.keras.datasets.mnist                                                #data set loading and scalling
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train[0:dataset_size,:,:]
x_test=x_test[0:dataset_size_t,:,:]
x_train=np.asarray(x_train).reshape(dataset_size,-1)
x_test=np.asarray(x_test).reshape(dataset_size_t,-1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = np.transpose(scaler.transform(x_train))
x_test=np.transpose(scaler.transform(x_test))

#x_train=2*(x_train > 0).astype(np.int)-1
#x_test=2*(x_test > 0).astype(np.int)-1
x_test_copy=x_test

#at.visualize_3d(P, W, in_index, t)

answer = input("Do you want to keep going y/n?")
if answer == "y":

    mode=0#grbm mode

    #layer 1
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.005#rbm rate
    rbmt.train_spatial_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[0])
    for i in range(0,700,100):
        plt.imshow(w[0][i,:].reshape(28,28))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)
    """ 
    mode=1#rbm mode
    
    #layer 2
    seconds = time.time()
    iterr_rbm=200
    epsilon=0.1#rbm rate
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    rbmt.train_spatial_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[1])
    print("layer 2")
    print("error", rbmv.error(x_test,b[1],c[1],w[1]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 1, scaler)
    
    #layer 3
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.05#rbm rate
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test=rbmt.sample_rbm_forward(x_test, c[1], w[1])
    rbmt.train_spatial_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[2])
    print("layer 3")
    print("error", rbmv.error(x_test,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 2, scaler)

    #layer 4
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.001#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test=rbmt.sample_rbm_forward(x_test, c[2], w[2])
    rbmt.train_spatial_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[3])
    print("layer 4")
    print("error", rbmv.error(x_test,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 3, scaler)
  
    #layer 5
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_4=rbmt.sample_rbm_forward(x_train_3, c[3], w[3])
    x_test=rbmt.sample_rbm_forward(x_test, c[3], w[3])
    rbmt.train_spatial_rbm(x_train_4, b[4],c[4],w[4], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[4])
    print("layer 5")
    print("error", rbmv.error(x_test,b[4],c[4],w[4]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 4, scaler)
    
    #layer 6
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_5=rbmt.sample_rbm_forward(x_train_4, c[4], w[4])
    x_test=rbmt.sample_rbm_forward(x_test, c[4], w[4])
    rbmt.train_spatial_rbm(x_train_5, b[5],c[5],w[5], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[5])
    print("layer 6")
    print("error", rbmv.error(x_test,b[5],c[5],w[5]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 5, scaler)
    
    #layer 7
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_6=rbmt.sample_rbm_forward(x_train_5, c[5], w[5])
    x_test=rbmt.sample_rbm_forward(x_test, c[5], w[5])
    rbmt.train_spatial_rbm(x_train_6, b[6],c[6],w[6], iterr_rbm, mode, epsilon, x_test, dataset_size, Wt[6])
    print("layer 7")
    print("error", rbmv.error(x_test,b[6],c[6],w[6]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 6, scaler)
      
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 0, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 1, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 2, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 3, scaler)
    """
    """   
    mode=0
    seconds = time.time()
    iterr_rbm=2000
    epsilon=0.001#rbm rate
    rbmt.train_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size)
    for i in range(w[0].shape[0]-10, w[0].shape[0]-5):
        plt.imshow(w[0][i,:].reshape(28,28))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)

    mode=1#rbm mode
    

    #layer 2
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.05#rbm rate
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 2")
    print("error", rbmv.error(x_test,b[1],c[1],w[1]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 1, scaler)
    
    #layer 3
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.01#rbm rate
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test=rbmt.sample_rbm_forward(x_test, c[1], w[1])
    rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 3")
    print("error", rbmv.error(x_test,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 2, scaler)
    
    #layer 4
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.001#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test=rbmt.sample_rbm_forward(x_test, c[2], w[2])
    rbmt.train_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 4")
    print("error", rbmv.error(x_test,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 3, scaler)
    
    #layer 5
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_4=rbmt.sample_rbm_forward(x_train_3, c[3], w[3])
    x_test=rbmt.sample_rbm_forward(x_test, c[3], w[3])
    rbmt.train_rbm(x_train_4, b[4],c[4],w[4], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 5")
    print("error", rbmv.error(x_test,b[4],c[4],w[4]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 4, scaler)
    
    #layer 6
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_5=rbmt.sample_rbm_forward(x_train_4, c[4], w[4])
    x_test=rbmt.sample_rbm_forward(x_test, c[4], w[4])
    rbmt.train_rbm(x_train_5, b[5],c[5],w[5], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 6")
    print("error", rbmv.error(x_test,b[5],c[5],w[5]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 5, scaler)
    
    #layer 7
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.01#rbm rate
    x_train_6=rbmt.sample_rbm_forward(x_train_5, c[5], w[5])
    x_test=rbmt.sample_rbm_forward(x_test, c[5], w[5])
    rbmt.train_rbm(x_train_6, b[6],c[6],w[6], iterr_rbm, mode, epsilon, x_test, dataset_size)
    print("layer 7")
    print("error", rbmv.error(x_test,b[6],c[6],w[6]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 6, scaler)
      
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 0, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 1, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 2, scaler)
    rbmv.gibbs_deep_sampling(x_test_copy[:,1:2], b, c, w, 3, scaler)
    """
elif answer == "n":
    print("ok")
else:
    print("Please enter y or n.")


