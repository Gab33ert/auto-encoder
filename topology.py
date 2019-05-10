# topographique auto encoder sturcture
import os
import tqdm
import voronoi
import numpy as np
np.set_printoptions(threshold=np.inf)


def build(n_cells, n_input_cells = 32, n_output_cells = 32, sparsity = 0.01, seed=0):
    """

    Parameters:
    -----------

    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator

    
    """

    #np.random.seed(seed)
    density    = np.ones((1000,1000))
    n=1000
    for i in range(n):
	    ii=i/(n-1)
	    density[:,i]=np.power((ii*(ii-2)+1)*np.ones((1,n)),6) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "autoencoder-rbm-MNIST.npy"#"CVT-%d-seed-%d.npy" % (n_cells,seed)
    if not os.path.exists(filename):
        cells_pos = np.zeros((n_cells,2))
        cells_pos[:,0] = np.random.uniform(0, 1000, n_cells)
        cells_pos[:,1] = np.random.uniform(0, 1000, n_cells)
        for i in tqdm.trange(75):
            _, cells_pos = voronoi.centroids(cells_pos, density, density_P, density_Q)
        np.save(filename, cells_pos)

    cells_pos = np.load(filename)
    cells_in  = np.argpartition(cells_pos, +n_input_cells, 0)[:+n_input_cells]
    cells_out = np.argpartition(cells_pos, -n_output_cells, 0)[-n_output_cells:]

    W=connect(cells_pos, n_input_cells, n_output_cells, d, sigma)

    return cells_pos/1000, W, cells_in[:,0], cells_out[:,0]
 
def build_3d(n, d, sigma):                                                     #n is the depth, d and sigma are connection caracteristics
    i=28#input image sze (MNIST is ims*ims)
    cells_pos=np.zeros((n*i*i,3))
    for z in range(n):
        for y in range(i):
            for x in range(i):
                #if np.random.random()<np.exp(-0.6*z):#((n-z)/n)**10:
                cells_pos[x+i*(y+i*z),0]=x*1000/(i-1)
                cells_pos[x+i*(y+i*z),1]=y*1000/(i-1)
                cells_pos[x+i*(y+i*z),2]=z*1000/(i-1)
    #cells_pos=np.concatenate((np.zeros((1,3)),cells_pos[np.argwhere(cells_pos[:,0]+cells_pos[:,1]+cells_pos[:,2]),:][:,0,:]))
    cells_pos[:,2]+=(1000/(i-1))*np.random.random(cells_pos[:,2].shape)
    cells_pos[0:783, 2]=(2500/(i-1))*np.ones(cells_pos[0:783, 2].shape)
    cells_pos[i*i-1 , 2]=cells_pos[i*i-2 , 2]
    return cells_pos/1000, connect_3d_sharp(cells_pos, d, sigma), np.arange(i*i)
"""
def build_3d(n, d, sigma):                                                     #n is the depth, d and sigma are connection caracteristics
    i=28#input image sze (MNIST is ims*ims)
    cells_pos=np.zeros((n*i*i,3))
    h=[28, 19, 15, 8, 6, 4]
    for z in range(n):
        j=h[z]#j=(28-6*z)
        for y in range(j):
            for x in range(j):
                #if np.random.random()<np.exp(-0.6*z):#((n-z)/n)**10:
                cells_pos[x+i*(y+i*z),0]=(x-0.5*(j-1))*1000/(i-1)+500
                cells_pos[x+i*(y+i*z),1]=(y-0.5*(j-1))*1000/(i-1)+500
                cells_pos[x+i*(y+i*z),2]=z*1000/(i-1)
    #cells_pos=np.concatenate((np.zeros((1,3)),cells_pos[np.argwhere(cells_pos[:,0]+cells_pos[:,1]+cells_pos[:,2]),:][:,0,:]))
    cells_pos[:,2]+=(1000/(i-1))*np.random.random(cells_pos[:,2].shape)
    #cells_pos[0:783, 2]=(500/(i-1))*np.ones(cells_pos[0:783, 2].shape)
    return cells_pos/1000, connect_3d_sharp(cells_pos, d, sigma), np.arange(i*i)
"""  
  


def connect(P, n_input_cells, n_output_cells, d, sigma):                       #build graph
    n = len(P)
    dP = P.reshape(1,n,2) - P.reshape(n,1,2)
    # Shifted Distances 
    D = np.hypot(dP[...,0]+d, dP[...,1])
    #W = np.zeros((n,n))
    W=np.where((np.random.uniform(0,1,(n,n)) < np.exp(-(D**2)/(2*sigma**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],0]>P[s[i,0],0]):
            W[s[i,0],s[i,1]]=0
    """
    for i in range(n):
        for j in range(n):
            if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))) & (P[i,0]<P[j,0]):
                W[j,i]=1
    """
    return W

def connect_3d(P, d, sigma):
    n = len(P)
    dP = P.reshape(1,n,3) - P.reshape(n,1,3)
    # Shifted Distances 
    D = np.hypot(dP[...,0], dP[...,1])
    D = np.hypot(D, dP[...,2]+d)
    #W = np.zeros((n,n))
    W=np.where((np.random.uniform(0,1,(n,n)) < np.exp(-(D**2)/(2*sigma**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],2]>=P[s[i,0],2]):
            W[s[i,0],s[i,1]]=0
    return W

def connect_3d_sharp(P, d, sigma):
    n = len(P)
    dP = P.reshape(1,n,3) - P.reshape(n,1,3)
    D = np.hypot(dP[...,0], dP[...,1])
    D = np.minimum(np.hypot(dP[...,0]+1000, dP[...,1]),D)
    D = np.minimum(np.hypot(dP[...,0], dP[...,1]+1000),D) 
    D = np.minimum(np.hypot(dP[...,0]+1000, dP[...,1]+1000),D)
    # Shifted Distances 
    D = np.hypot(0.1*D, dP[...,2]+d)
    #W = np.zeros((n,n))
    W=np.where((D-sigma)<0, 1 , 0)#np.where((np.random.uniform(0,1,(n,n)) < np.exp((-np.power(np.maximum(0,D-4*sigma),2))/(2*(sigma/2)**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],2]>=P[s[i,0],2]):
            W[s[i,0],s[i,1]]=0
    return W



def abstract_layer(in_index, W, t):                                            #unfold the total connection matrix into layer by layer connection matrix
    index=in_index
    Wt=[]
    for i in range(t-1):
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_new=np.asarray([idx for idx, v in enumerate(x) if v])
        Wt.append(W[index_new[:, None], index])
        index=index_new
        print(Wt[i].shape)
    return Wt

def abstract_layer_local_backward_restriction(in_index, W, t, n_min):           #unfold the total connection matrix into layer by layer connection matrix
    index=in_index                                                               #at each layers neurons are connected to at least n_min neurons
    Wt=[]
    index_list=[]
    index_list.append(in_index)
    for i in range(t-1):
        print(i)
        x=np.zeros(W.shape[0])
        for j in index:x[j]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index_new=np.asarray([idx for idx, v in enumerate(x) if v])
        index_new=index_new[np.argwhere(np.sum(W[index_new[:, None], index], axis=1) >= n_min)]#/index.shape[0]
        index_new=index_new.reshape((index_new.shape[0],))
        Wt.append(W[index_new[:, None], index])
        index_list.append(index_new)
        index=index_new
        print(Wt[i].shape)
    return Wt, index_list

def abstract_layer_local_forward_restriction(in_index, W, t, n_min):           #unfold the total connection matrix into layer by layer connection matrix
   index=in_index                                                               #at each layers neurons are connected to at least n_min neurons
   Wt=[]
   x=np.zeros(W.shape[0])
   for j in index:x[j]=1
   x=W.dot(x)
   x=(x > 0).astype(int)
   index_new=np.asarray([idx for idx, v in enumerate(x) if v])
   index_new=index_new.reshape((index_new.shape[0],))
   index_old=index
   index=index_new
   for i in range(1,t):
       x=np.zeros(W.shape[0])
       for j in index:x[j]=1
       x=W.dot(x)
       x=(x > 0).astype(int)
       index_new=np.asarray([idx for idx, v in enumerate(x) if v])
       index=index[np.argwhere(np.sum(W[index_new,:][:, index], axis=0) >= n_min)]#/index.shape[0]
       Wt.append(W[index.reshape((index.shape[0],)),:][:, index_old])
       print(Wt[i-1].shape)
       x=np.zeros(W.shape[0])
       for j in index:x[j]=1
       x=W.dot(x)
       x=(x > 0).astype(int)
       index_new=np.asarray([idx for idx, v in enumerate(x) if v])
       index_new=index_new.reshape((index_new.shape[0],))
       index_old=index.reshape((index.shape[0],))
       index=index_new
   return Wt

def abstract_layer_restriction(Wt, n, rate):                                           #choose n unit in final layer with high connection to input
    k=0
    j=500
    le=len(Wt)

    index=[]
    
    while k<n:
        if j==Wt[le-1].shape[0]:
            print("we couldn't find "+str(n)+" final unit")
            break
        out=np.zeros((1,Wt[le-1].shape[0]))
        out[0,j]=1
        for i in range(le):
            out=out.dot(Wt[le-1-i])
            out=(out > 0).astype(int)
        if np.sum(out)/out.shape[1]> rate:
            k+=1
            index.append(j)
        j+=1
    
    print("index", index)
    
    out=np.zeros((1,Wt[le-1].shape[0]))
    out[0][index]=1
    for i in range(le): 
        out_n=out.dot(Wt[le-1-i])
        out_n=(out_n > 0).astype(int)
        index_n=np.argwhere(out_n==1)[:,1]
        Wt[le-1-i]=Wt[le-1-i][index,:][:,index_n]
        out=out_n
        index=index_n
        print(Wt[le-1-i].shape)
    return Wt
    
