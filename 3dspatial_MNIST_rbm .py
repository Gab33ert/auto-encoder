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
from skimage.transform import resize

def MNIST_red(x):
    ind=np.ones(14)
    for i in range(14):
        ind[i]=2*i
    ind=ind.astype(int)
    return x[:,ind, :][:,:, ind]
    

 

#global variable
size=784
n_cell=1600
dataset_size=10000
dataset_size_t=300
t=5
sigma=16#4
ims=28
d=27#260
alpha=0.0005#backprop rate

epsilon=0.001#rbm rate


P, W,in_index =top.build_3d(7, d, sigma)
#Wt=top.abstract_layer(in_index, W, t)#_local_backward_restriction
Wt, index_list=top.abstract_layer_local_backward_restriction(in_index, W, t,16)
at.visualize_abstract_3d(P, W, index_list, t)
at.connection_forward_0(Wt)
at.connection_backward_rate(Wt)
at.connection_forward_rate(Wt)
#at.analyze_topology_back(Wt,4)
#Wt=top.abstract_layer_restriction(Wt, 10, 0.93)
#at.connection_backward_rate(Wt)
#at.analyze_topology_back(Wt,4)
#for i in range(t-1,0,-1):
    #at.analyze_topology(Wt, i)
#for i in range(t-1,0,-1):
    #at.analyze_topology_froward(Wt,i)
for i in range(t-1,0,-1):
    at.analyze_topology_back(Wt,i)
#at.connection_backward_rate(Wt)
    
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

#bt = np.transpose(resize(np.transpose(x_train), (14, 14), anti_aliasing=True))
#x_train=MNIST_red(x_train)
#x_test=MNIST_red(x_test)

x_train=np.asarray(x_train).reshape(dataset_size,-1)
x_test=np.asarray(x_test).reshape(dataset_size_t,-1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = np.transpose(scaler.transform(x_train))
x_test=np.transpose(scaler.transform(x_test))

#x_train=2*(x_train > 0).astype(np.int)-1
#x_test=2*(x_test > 0).astype(np.int)-1


#at.visualize_3d(P, W, in_index, t)
mode=[0,1]                                                                      #first digit 0/1 grbm/rbm, second digit 0 no sparsity p!=0 sparsity rate parameter
answer = input("Do you want to keep going y/n?")
if answer == "y":

    mode[0]=0#grbm mode

    b[0]=np.random.randn(Wt[0].shape[1],1)
    c[0]=np.zeros(c[0].shape)    
    w[0]=copy.deepcopy(Wt[0])
    w[0]=w[0]*np.random.normal(0, 0.01, w[0].shape)

    #layer 1
    seconds = time.time()
    iterr_rbm=3000
    epsilon=0.01#rbm rate
    rbmt.train_spatial_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size, 1, 50, Wt[0])
    for i in range(0,w[0].shape[0]-1,w[0].shape[0]//5):
        plt.imshow(w[0][i,:].reshape(ims,ims))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)

    mode[0]=1#rbm mode
    
    b[1]=np.zeros(b[1].shape)
    c[1]=np.zeros(c[1].shape)    
    w[1]=copy.deepcopy(Wt[1])
    w[1]=w[1]*np.random.normal(0, 0.01, w[1].shape)
    
    #layer 2
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.05#rbm rate
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    rbmt.train_spatial_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test_1, dataset_size, 1, 50, Wt[1])
    print("layer 2")
    print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)

    b[2]=np.zeros(b[2].shape)
    c[2]=np.zeros(c[2].shape)
    w[2]=copy.deepcopy(Wt[2])
    w[2]=w[2]*np.random.normal(0, 0.01, w[2].shape) 
    #layer 3
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.01#0.05#rbm rate
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
    rbmt.train_spatial_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test_2, dataset_size, 1, 50, Wt[2])
    print("layer 3")
    print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)

    b[3]=np.zeros(b[3].shape)
    c[3]=np.zeros(c[3].shape)
    w[3]=copy.deepcopy(Wt[3])
    w[3]=w[3]*np.random.normal(0, 0.01, w[3].shape) 
    #layer 4
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.1#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
    rbmt.train_spatial_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test_3, dataset_size, 1, 50, Wt[3])
    print("layer 4")
    print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
    
    
    for i in range(4):
        plt.hist(w[i][np.nonzero(w[i])], 100)
        plt.show()

    x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
    n=10
    pi=0
    p=50#x_test_4.shape[0]
    x=np.zeros((10*n, p))
    for j in range(10):                                                         #affiche l'encodage
        q=0
        i=0
        while (q<n and i<300):
            if y_test[i]==j:
                x[q+n*j, 0:p]=0.5*(x_test_4[pi:pi+p, i]+1)
                q+=1
            i+=1
    plt.imshow(np.transpose(x))
        
    """

    b[0]=np.random.randn(Wt[0].shape[1],1)
    c[0]=np.zeros(c[0].shape)
    w[0]=np.ones(w[0].shape)*np.random.normal(0, 0.01, w[0].shape)
    
    mode=0
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.0002#rbm rate
    rbmt.train_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size, 1, 50)
    for i in range(w[0].shape[0]-10, w[0].shape[0]-5):
        plt.imshow(w[0][i,:].reshape(ims,ims))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)

    mode=1#rbm mode

    #layer 2

    b[1]=np.zeros(b[1].shape)
    c[1]=np.zeros(c[1].shape)
    w[1]=np.ones(w[1].shape)*np.random.normal(0, 0.01, w[1].shape)

    seconds = time.time()
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    iterr_rbm=2000#15000
    epsilon=0.004#0.02#rbm rate
    rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test_1, dataset_size, 1, 50)
    print("time",time.time()-seconds)

    #iterr_rbm=3000
    #epsilon=0.002
    #rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size, 10, 200)
    #print("time",time.time()-seconds)
    #iterr_rbm=18000
    #epsilon=0.0001
    #rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size, 50, 400)
 
    print("layer 2")
    print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))

    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)

    b[2]=np.zeros(b[2].shape)
    c[2]=np.zeros(c[2].shape)
    w[2]=np.ones(w[2].shape)*np.random.normal(0, 0.01, w[2].shape)

 
    #layer 3
    seconds = time.time()
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
    iterr_rbm=2000
    epsilon=0.005#rbm rate       #test it with 0.05
    rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test_2, dataset_size, 1, 200)

    #iterr_rbm=100000
    #epsilon=0.00005#rbm rate
    #rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test, dataset_size, 10, 200)

    print("layer 3")
    print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)

    #layer 4
    
    b[3]=np.zeros(b[3].shape)
    c[3]=np.zeros(c[3].shape)
    w[3]=np.ones(w[3].shape)*np.random.normal(0, 0.01, w[3].shape)

    
    seconds = time.time()
    iterr_rbm=10000
    epsilon=0.1#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
    rbmt.train_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test_3, dataset_size, 1, 200)
    print("layer 4")
    print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
    
    x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
    for j in range(10):
        n=30
        q=0
        i=0
        print(str(j))
        x=np.zeros(x_test_4.shape[0])
        while (q<n and i<300):
            if y_test[i]==j:
                x+=0.5*(x_test_4[:,i]+1)
                q+=1
            i+=1
        print((10*x/n)//1)

    n=10
    pi=0
    p=50#x_test_4.shape[0]
    x=np.zeros((10*n, p))
    for j in range(10):                                                         #affiche l'encodage
        q=0
        i=0
        while (q<n and i<300):
            if y_test[i]==j:
                x[q+n*j, 0:p]=0.5*(x_test_4[pi:pi+p, i]+1)
                q+=1
            i+=1
    plt.imshow(np.transpose(x))
        
        

    
    for i in range(4):
        plt.hist(w[i])
        plt.show()
    """
elif answer == "n":
    print("ok")
else:
    print("Please enter y or n.")


"""
with parameter sigma=4,d=27,  restriction=32/33
0.1*D
and size 28*28

    mode=0#grbm mode

    b[0]=np.random.randn(Wt[0].shape[1],1)
    c[0]=np.zeros(c[0].shape)    
    w[0]=copy.deepcopy(Wt[0])
    w[0]=w[0]*np.random.normal(0, 0.01, w[0].shape)

    #layer 1
    seconds = time.time()
    iterr_rbm=3000
    epsilon=0.001#rbm rate
    rbmt.train_spatial_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size, 1, 50, Wt[0])
    for i in range(0,w[0].shape[0]-1,w[0].shape[0]//5):
        plt.imshow(w[0][i,:].reshape(28,28))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)

    mode=1#rbm mode
    
    b[1]=np.zeros(b[1].shape)
    c[1]=np.zeros(c[1].shape)    
    w[1]=copy.deepcopy(Wt[1])
    w[1]=w[1]*np.random.normal(0, 0.01, w[1].shape)
    
    #layer 2
    seconds = time.time()
    iterr_rbm=200
    epsilon=0.5#rbm rate
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    rbmt.train_spatial_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test_1, dataset_size, 1, 50, Wt[1])
    print("layer 2")
    print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)

    b[2]=np.zeros(b[2].shape)
    c[2]=np.zeros(c[2].shape)
    w[2]=copy.deepcopy(Wt[2])
    w[2]=w[2]*np.random.normal(0, 0.01, w[2].shape) 
    #layer 3
    seconds = time.time()
    iterr_rbm=1000
    epsilon=0.3#0.05#rbm rate
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
    rbmt.train_spatial_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test_2, dataset_size, 1, 50, Wt[2])
    print("layer 3")
    print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)

    b[3]=np.zeros(b[3].shape)
    c[3]=np.zeros(c[3].shape)
    w[3]=copy.deepcopy(Wt[3])
    w[3]=w[3]*np.random.normal(0, 0.01, w[3].shape) 
    #layer 4
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.001#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
    rbmt.train_spatial_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test_3, dataset_size, 1, 50, Wt[3])
    print("layer 4")
    print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
    
    
    
    
    


    b[0]=np.random.randn(Wt[0].shape[1],1)
    c[0]=np.zeros(c[0].shape)
    w[0]=np.ones(w[0].shape)*np.random.normal(0, 0.01, w[0].shape)
    
    mode=0
    seconds = time.time()
    iterr_rbm=5000
    epsilon=0.0002#rbm rate
    rbmt.train_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, x_test, dataset_size, 1, 50)
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

    b[1]=np.zeros(b[1].shape)
    c[1]=np.zeros(c[1].shape)
    w[1]=np.ones(w[1].shape)*np.random.normal(0, 0.01, w[1].shape)

    seconds = time.time()
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    iterr_rbm=2000#15000
    epsilon=0.004#0.02#rbm rate
    rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test_1, dataset_size, 1, 50)
    print("time",time.time()-seconds)

    #iterr_rbm=3000
    #epsilon=0.002
    #rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size, 10, 200)
    #print("time",time.time()-seconds)
    #iterr_rbm=18000
    #epsilon=0.0001
    #rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, x_test, dataset_size, 50, 400)
 
    print("layer 2")
    print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))

    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)

    b[2]=np.zeros(b[2].shape)
    c[2]=np.zeros(c[2].shape)
    w[2]=np.ones(w[2].shape)*np.random.normal(0, 0.01, w[2].shape)

 
    #layer 3
    seconds = time.time()
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
    iterr_rbm=2000
    epsilon=0.005#rbm rate       #test it with 0.05
    rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test_2, dataset_size, 1, 200)

    #iterr_rbm=100000
    #epsilon=0.00005#rbm rate
    #rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, x_test, dataset_size, 10, 200)

    print("layer 3")
    print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)

    #layer 4
    
    b[3]=np.zeros(b[3].shape)
    c[3]=np.zeros(c[3].shape)
    w[3]=np.ones(w[3].shape)*np.random.normal(0, 0.01, w[3].shape)

    
    seconds = time.time()
    iterr_rbm=10000
    epsilon=0.1#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
    rbmt.train_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, x_test_3, dataset_size, 1, 200)
    print("layer 4")
    print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
    
    x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
    for j in range(10):
        n=30
        q=0
        i=0
        print(str(j))
        x=np.zeros(x_test_4.shape[0])
        while (q<n and i<300):
            if y_test[i]==j:
                x+=0.5*(x_test_4[:,i]+1)
                q+=1
            i+=1
        print((10*x/n)//1)
    
    
    for i in range(4):
        plt.hist(w[i])
        plt.show()
    
"""
