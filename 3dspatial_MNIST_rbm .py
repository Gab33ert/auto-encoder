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
import pickle

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
    
class store(object): 
    def __init__(self, build_height=0, build_d=0, build_sigma=0, W=0, Wt=0, P=0, index_list=0, Spatial=0, epsilon=0, alpha=0, mode=0, iterr_rbm=0, w=0, b=0, c=0, fe=0, ttt=0):
        self.build_height = build_height#build length
        self.build_d = build_d#connection length
        self.build_sigma = build_sigma#connection width
        self.W = W#complete connection matrix
        self.Wt = Wt#layered connection matrix
        self.P = P#position array
        self.index_list = index_list#list of index of layers suposed to be 5
        
        self.Spatial = Spatial#==True if spatial connection
        self.epsilon = epsilon#list of all learning rate (supposed to be 4)
        self.alpha = alpha#list of sparsity learning rate
        self.mode = mode#list of list(first element 0/1->gaussianRbm/rbm, second element, if mode[i][1]==0 no sparsity, otherwise mode[i][1] is the target sparsity)
        self.iterr_rbm = iterr_rbm#list of iterration for rbm training
        
        self.w = w#learned weight
        self.b = b#learned backward bias
        self.c = c#learned forward bias
        
        self.fe = fe#training free energy
        self.ttt = ttt

 

#global variable
size=784
n_cell=1600
dataset_size=10000
dataset_size_t=300
t=4
sigma=16#4
ims=28
d=27#260
alpha=0.0005#backprop rate
height=7
epsilon=[0]*4#rbm rate
fe=[]
ttt=[]
Spatial=False


P, W,in_index =top.build_3d(height, d, sigma)
#Wt=top.abstract_layer(in_index, W, t)#_local_backward_restriction
Wt, index_list=top.abstract_layer_local_backward_restriction(in_index, W, t, 1)#385)
at.visualize_abstract_3d(P, W, index_list, t)
at.degree_distribution(Wt)
#at.connection_forward_0(Wt)
#at.connection_backward_rate(Wt)
#at.connection_forward_rate(Wt)
#at.analyze_topology_back(Wt,4)
#Wt=top.abstract_layer_restriction(Wt, 10, 0.93)
#at.connection_backward_rate(Wt)
#at.analyze_topology_back(Wt,4)
#for i in range(t-1,0,-1):
    #at.analyze_topology(Wt, i)
#for i in range(t-1,0,-1):
    #at.analyze_topology_froward(Wt,i)
#for i in range(t-1,0,-1):
#    at.analyze_topology_back(Wt,i)
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


#at.visualize_3d(P, W, in_index, t)
answer = input("Do you want to train y/n?")
if answer == "y":
    if Spatial:
        epsilon=[0.01, 0.05, 0.01, 0.07] #RBMs learning rate
        alpha=[1, 0.1, 0.01, 0.05]      #Sparsity learning rate ponderation
        mode=[[0, 0], [1, 0], [1, 0], [1, 0.1]] #mode[i][0] 0 for grbm 1 for rbm   mode[i][1] 0 if no sparsity if !=0 mode[i][1]=target sparsity
        iterr_rbm=[3000, 1000, 3000, 1000]
            
        b[0]=np.random.randn(Wt[0].shape[1],1)
        c[0]=np.zeros(c[0].shape)    
        w[0]=copy.deepcopy(Wt[0])
        w[0]=w[0]*np.random.normal(0, 0.01, w[0].shape)

        #layer 1
        seconds = time.time()
        fe.append(rbmt.train_spatial_rbm(x_train, b[0], c[0], w[0], iterr_rbm[0], mode[0], epsilon[0], alpha[0], x_test, dataset_size, 1, 50, Wt[0]))
        ttt.append(time.time()-seconds)
        print("time",time.time()-seconds)
        print("error", rbmv.error(x_test,b[0],c[0],w[0]))
        for i in range(0,w[0].shape[0]-1,w[0].shape[0]//5):
            plt.imshow(w[0][i,:].reshape(ims,ims))
            plt.colorbar()
            plt.show()
        rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode[0])
        rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode[0])
        rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode[0])
    
        x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
        n=20
        pi=0
        p=200#x_test_4.shape[0]
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_1[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
    
        
        b[1]=np.zeros(b[1].shape)
        c[1]=np.zeros(c[1].shape)    
        w[1]=copy.deepcopy(Wt[1])
        w[1]=w[1]*np.random.normal(0, 0.01, w[1].shape)
        
        #layer 2
        x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
        x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
        seconds = time.time()
        fe.append(rbmt.train_spatial_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm[1], mode[1], epsilon[1], alpha[1], x_test_1, dataset_size, 1, 50, Wt[1]))
        ttt.append(time.time()-seconds)
        print("layer 2")
        print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))
        print("time",time.time()-seconds)
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)
    
        x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
        n=20
        pi=0
        p=200#x_test_4.shape[0]
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_2[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
    


        b[2]=np.zeros(b[2].shape)
        c[2]=np.zeros(c[2].shape)
        w[2]=copy.deepcopy(Wt[2])
        w[2]=w[2]*np.random.normal(0, 0.01, w[2].shape) 
        #layer 3
        x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
        x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
        seconds = time.time()
        fe.append(rbmt.train_spatial_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm[2], mode[2], epsilon[2], alpha[2], x_test_2, dataset_size, 1, 50, Wt[2]))
        ttt.append(time.time()-seconds)
        print("layer 3")
        print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
        print("time",time.time()-seconds)
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)
    
        x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
        n=20
        pi=0
        p=min(200, x_test_3.shape[0])#x_test_4.shape[0]
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_3[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
    
        """
        b[3]=np.zeros(b[3].shape)
        c[3]=np.zeros(c[3].shape)
        w[3]=copy.deepcopy(Wt[3])
        w[3]=w[3]*np.random.normal(0, 0.01, w[3].shape) 
        #layer 4
        x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
        x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
        seconds = time.time()
        fe.append(rbmt.train_spatial_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm[3], mode[3], epsilon[3], alpha[3], x_test_3, dataset_size, 1, 50, Wt[3]))
        ttt.append(time.time()-seconds)
        print("layer 4")
        print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
        print("time",time.time()-seconds)
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
        
        
        for i in range(4):
            plt.hist(w[i][np.nonzero(w[i])], 100)
            plt.show()
    
        x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
        n=20
        pi=0
        p=min(200, x_test_4.shape[0])#x_test_4.shape[0]
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_4[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        """
    
    else:
        epsilon=[0.001, 0.001, 0.001, 0.001]
        alpha=[5, 0.5, 0.05, 8]
        mode=[[0, 0.1], [1, 0.1], [1, 0.1], [1, 0]]#[[0, 0.1], [1, 0.1], [1, 0.1], [1, 0.2]]
        iterr_rbm=[6000, 2000, 2000, 10]
        
        b[0]=np.random.randn(Wt[0].shape[1],1)
        c[0]=np.zeros(c[0].shape)
        w[0]=np.ones(w[0].shape)*np.random.normal(0, 0.01, w[0].shape)
        
        seconds = time.time()
        fe.append(rbmt.train_rbm(x_train, b[0], c[0], w[0], iterr_rbm[0], mode[0], epsilon[0], alpha[0], x_test, dataset_size, 1, 50))
        ttt.append(time.time()-seconds)
        for i in range(w[0].shape[0]-10, w[0].shape[0]-5):
            plt.imshow(w[0][i,:].reshape(ims,ims))
            plt.colorbar()
            plt.show()
        print("error", rbmv.error(x_test,b[0],c[0],w[0]))
        print("time",time.time()-seconds)
        rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode[0])
        rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode[0])
        rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode[0])
     
        

    
        #layer 2
    
        b[1]=np.zeros(b[1].shape)
        c[1]=np.zeros(c[1].shape)
        w[1]=np.ones(w[1].shape)*np.random.normal(0, 0.01, w[1].shape)
    
        x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
        x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
        seconds = time.time()
        fe.append(rbmt.train_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm[1], mode[1], epsilon[1], alpha[1], x_test_1, dataset_size, 1, 50))
        ttt.append(time.time()-seconds)
        print("time",time.time()-seconds)
    
     
        print("layer 2")
        print("error", rbmv.error(x_test_1,b[1],c[1],w[1]))
    
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 1, scaler)
    
        b[2]=np.zeros(b[2].shape)
        c[2]=np.zeros(c[2].shape)
        w[2]=np.ones(w[2].shape)*np.random.normal(0, 0.01, w[2].shape)
    
     
        #layer 3
        x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
        x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
        seconds = time.time()
        fe.append(rbmt.train_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm[2], mode[2], epsilon[2], alpha[2], x_test_2, dataset_size, 1, 200))
        ttt.append(time.time()-seconds)
        print("layer 3")
        print("error", rbmv.error(x_test_2,b[2],b[3],w[2]))
        print("time",time.time()-seconds)
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 2, scaler)
    
        """
    
        #layer 4
        
        b[3]=np.zeros(b[3].shape)
        c[3]=np.zeros(c[3].shape)
        w[3]=np.ones(w[3].shape)*np.random.normal(0, 0.01, w[3].shape)
    
    
        x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
        x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
        seconds = time.time()
        fe.append(rbmt.train_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm[3], mode[3], epsilon[3], alpha[3], x_test_3, dataset_size, 1, 200))
        ttt.append(time.time()-seconds)
        print("layer 4")
        print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
        print("time",time.time()-seconds)
        rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
        
        x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
        """
        x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
        n=10
        pi=0
        p=min(200,x_test_1.shape[0])#x_test_4.shape[0]
        
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_1[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
        p=min(200,x_test_2.shape[0])
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_2[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
        p=min(200,x_test_3.shape[0])
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_3[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
    
        
        for i in range(4):
            plt.hist(w[i])
            plt.show()
            
        """  
        #one layer sparse encoding
        b[0]=np.random.randn(Wt[0].shape[1],1)
        c[0]=np.zeros(c[0].shape)
        w[0]=np.ones(w[0].shape)*np.random.normal(0, 0.01, w[0].shape)
        
        alpha=8
        mode[1]=0.1
        mode[0]=0
        seconds = time.time()
        iterr_rbm=5000
        epsilon=0.002#rbm rate
        rbmt.train_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, alpha, x_test, dataset_size, 1, 50)
        for i in range(w[0].shape[0]-10, w[0].shape[0]-5):
            plt.imshow(w[0][i,:].reshape(ims,ims))
            plt.colorbar()
            plt.show()
        print("error", rbmv.error(x_test,b[0],c[0],w[0]))
        print("time",time.time()-seconds)
        rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
        rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
        rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)
        
        x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
        n=20
        pi=0
        p=100#x_test_4.shape[0]
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_1[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        """
        
    answer = input("Do you want to save it y/n?")
    if answer == "y":
        memory = store(height, d, sigma, W, Wt, P, index_list, Spatial, epsilon, alpha, mode, iterr_rbm, w, b, c, fe, ttt)
        with open('no_restriction_sparse_dbn_non_spatial.pkl', 'wb') as output:
            pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)
    elif answer == "n":
        print("ok")
    else:
        print("Please enter y or n.")
    
    
elif answer == "n":
    print("ok")
else:
    print("Please enter y or n.")
    
    
strl=['no_restriction_sparse_dbn_non_spatial.pkl', 'no_restriction_spatial_dbn.pkl']#['sparse_dbn_non_spatial.pkl', 'no_restriction_sparse_dbn_non_spatial.pkl', 'no_restriction_sparse_spatial_dbn.pkl', 'no_restriction_spatial_dbn.pkl']
answer = input("Do you want to load y/n?")
if answer == "y":#load already trained network parameter and plot reconstruction.
    for i in range(6):
        plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(x_test[:,1+i:2+i]))).reshape(28,28), vmin=-100, vmax= 400)
        plt.show()
    fig = plt.figure()
    for i in range(10):
            if i<5:fig.add_subplot(6,5,1+i)
            else:fig.add_subplot(6,5,11+i)
            plt.axis('off')
            plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(x_test[:,i]))).reshape(ims,ims), vmin=-100, vmax= 400)
    k=0
    for st in strl:
        print(st)
        with open(st, 'rb') as data:
            memory = pickle.load(data)
            w=memory.w 
            b=memory.b 
            c=memory.c 
            
            print("alpha", memory.alpha)
            print("mode", memory.mode)
        for i in range(10):
            n=2
            h=rbmt.sample_rbm_forward(x_test[:,i:i+1], c[0], w[0])
            for j in range(n):
                h=rbmt.sample_rbm_forward(h, c[j+1], w[j+1])
            for j in range(n):
                h=rbmt.sample_rbm_backward(h, b[n-j], w[n-j])
            h=rbmt.sample_grbm_backward(h, b[0], w[0])
            if i<5:fig.add_subplot(6,5,6+i+k)
            else:fig.add_subplot(6,5,16+i+k)
            plt.axis('off')
            plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(h))).reshape(ims,ims), vmin=-100, vmax= 400)
        k+=5
                
        
        x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
        x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
        x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
        
        """
        n=10
        pi=0
        p=min(200,x_test_1.shape[0])#x_test_4.shape[0]
        
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_1[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
        p=min(200,x_test_2.shape[0])
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_2[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
        p=min(200,x_test_3.shape[0])
        x=np.zeros((10*n, p))
        for j in range(10):                                                         #affiche l'encodage
            q=0
            i=0
            while (q<n and i<300):
                if y_test[i]==j:
                    x[q+n*j, 0:p]=0.5*(x_test_3[pi:pi+p, i]+1)
                    q+=1
                i+=1
        fig, ax = plt.subplots(figsize=(90, 10))
        ax.imshow(x)
        plt.show()
        
        for i in range(3):
            plt.plot(memory.fe[i][0], memory.fe[i][1], label="layer "+str(i+1))
        plt.legend()
        plt.show()
        
        plt.hist(np.mean((x_test_1+1)/2, axis=1), bins=30)
        plt.show() 
        plt.hist(np.mean((x_test_2+1)/2, axis=1), bins=30)
        plt.show()     
        plt.hist(np.mean((x_test_3+1)/2, axis=1), bins=30)
        plt.show()         
        """
        
    #plt.savefig("mnistDbn.pdf")
    plt.show()
        
elif answer == "n":
    print("ok")
else:
    print("Please enter y or n.")
    
    
    
"""#works without sparsity but spatial
    mode[0]=0#grbm mode
    alpha=0.5

    b[0]=np.random.randn(Wt[0].shape[1],1)
    c[0]=np.zeros(c[0].shape)    
    w[0]=copy.deepcopy(Wt[0])
    w[0]=w[0]*np.random.normal(0, 0.01, w[0].shape)

    #layer 1
    seconds = time.time()
    iterr_rbm=3000
    epsilon=0.01#rbm rate
    rbmt.train_spatial_rbm(x_train, b[0], c[0], w[0], iterr_rbm, mode, epsilon, alpha, x_test, dataset_size, 1, 50, Wt[0])
    for i in range(0,w[0].shape[0]-1,w[0].shape[0]//5):
        plt.imshow(w[0][i,:].reshape(ims,ims))
        plt.colorbar()
        plt.show()
    print("error", rbmv.error(x_test,b[0],c[0],w[0]))
    print("time",time.time()-seconds)
    rbmv.gibbs_sampling(x_test[:,0:1], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,1:2], b[0],c[0],w[0], 2, scaler, mode)
    rbmv.gibbs_sampling(x_test[:,20:21], b[0],c[0],w[0], 2, scaler, mode)

    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    n=20
    pi=0
    p=200#x_test_4.shape[0]
    x=np.zeros((10*n, p))
    for j in range(10):                                                         #affiche l'encodage
        q=0
        i=0
        while (q<n and i<300):
            if y_test[i]==j:
                x[q+n*j, 0:p]=0.5*(x_test_1[pi:pi+p, i]+1)
                q+=1
            i+=1
    fig, ax = plt.subplots(figsize=(90, 10))
    ax.imshow(x)


    mode[0]=1#rbm mode
    
    b[1]=np.zeros(b[1].shape)
    c[1]=np.zeros(c[1].shape)    
    w[1]=copy.deepcopy(Wt[1])
    w[1]=w[1]*np.random.normal(0, 0.01, w[1].shape)
    
    #layer 2
    seconds = time.time()
    iterr_rbm=500#2000
    epsilon=0.05#rbm rate
    x_train_1=rbmt.sample_rbm_forward(x_train, c[0], w[0])
    x_test_1=rbmt.sample_rbm_forward(x_test, c[0], w[0])
    rbmt.train_spatial_rbm(x_train_1, b[1],c[1],w[1], iterr_rbm, mode, epsilon, alpha, x_test_1, dataset_size, 1, 50, Wt[1])
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
    iterr_rbm=3000#5000
    epsilon=0.01#0.05#rbm rate
    x_train_2=rbmt.sample_rbm_forward(x_train_1, c[1], w[1])
    x_test_2=rbmt.sample_rbm_forward(x_test_1, c[1], w[1])
    rbmt.train_spatial_rbm(x_train_2, b[2],c[2],w[2], iterr_rbm, mode, epsilon, alpha, x_test_2, dataset_size, 1, 50, Wt[2])
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
    iterr_rbm=1000#5000
    epsilon=0.1#rbm rate
    x_train_3=rbmt.sample_rbm_forward(x_train_2, c[2], w[2])
    x_test_3=rbmt.sample_rbm_forward(x_test_2, c[2], w[2])
    rbmt.train_spatial_rbm(x_train_3, b[3],c[3],w[3], iterr_rbm, mode, epsilon, alpha, x_test_3, dataset_size, 1, 50, Wt[3])
    print("layer 4")
    print("error", rbmv.error(x_test_3,b[3],c[3],w[3]))
    print("time",time.time()-seconds)
    rbmv.gibbs_deep_sampling(x_test[:,1:2], b, c, w, 3, scaler)
    
    
    for i in range(4):
        plt.hist(w[i][np.nonzero(w[i])], 100)
        plt.show()

    x_test_4=rbmt.sample_rbm_forward(x_test_3, c[3], w[3])
    n=20
    pi=0
    p=200#x_test_4.shape[0]
    x=np.zeros((10*n, p))
    for j in range(10):                                                         #affiche l'encodage
        q=0
        i=0
        while (q<n and i<300):
            if y_test[i]==j:
                x[q+n*j, 0:p]=0.5*(x_test_4[pi:pi+p, i]+1)
                q+=1
            i+=1
    fig, ax = plt.subplots(figsize=(90, 10))
    ax.imshow(x)
        
    
"""
