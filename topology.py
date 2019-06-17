# Function for creating the DBN topology
import numpy as np
np.set_printoptions(threshold=np.inf)


def build_3d(n, d, sigma):                                                     #n is the depth, d and sigma are connection caracteristics
    i=28#input image sze (MNIST is ims*ims)
    cells_pos=np.zeros((n*i*i,3))
    for z in range(n):
        for y in range(i):
            for x in range(i):
                if z!=0:
                    if np.random.random()<np.exp(-z):#(4*np.exp(-4*z)+1)/5:#((n-z)/n)**10:#0.15*np.exp(-0.1*z):#
                        cells_pos[x+i*(y+i*z),0]=x*1000/(i-1)
                        cells_pos[x+i*(y+i*z),1]=y*1000/(i-1)
                        cells_pos[x+i*(y+i*z),2]=z*1000/(i-1)
                else:
                    cells_pos[x+i*(y+i*z),0]=x*1000/(i-1)
                    cells_pos[x+i*(y+i*z),1]=y*1000/(i-1)
                    cells_pos[x+i*(y+i*z),2]=1
    cells_pos=np.concatenate((np.zeros((1,3)),cells_pos[np.argwhere(cells_pos[:,0]+cells_pos[:,1]+cells_pos[:,2]),:][:,0,:]))
    for y in range(i):
        for x in range(i):
            cells_pos[x+i*y,2]=0
    cells_pos[:,2]+=(1000/(i-1))*np.random.random(cells_pos[:,2].shape)
    #cells_pos[0:783, 2]=(2500/(i-1))*np.ones(cells_pos[0:783, 2].shape)
    #cells_pos[i*i-1 , 2]=cells_pos[i*i-2 , 2]
    return cells_pos/1000, connect_3d_sharp(cells_pos, d, sigma), np.arange(i*i)
  


def connect_3d(P, d, sigma):#connect using exponential scheme takes positions P, and parameters d, sigma as entry. Returns the connection matrix W with 0 and 1 
    n = len(P)
    dP = P.reshape(1,n,3) - P.reshape(n,1,3)
    # Shifted Distances 
    D = np.hypot(dP[...,0], dP[...,1])
    D = np.hypot(0.01*D, dP[...,2]+d)
    #W = np.zeros((n,n))
    W=np.where((np.random.uniform(0,1,(n,n)) < np.exp(-(D**2)/(2*sigma**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],2]>=P[s[i,0],2]):
            W[s[i,0],s[i,1]]=0
    return W

def connect_3d_sharp(P, d, sigma):#connect using sharp ellipse scheme takes positions P, and parameters d, sigma as entry. Returns the connection matrix W with 0 and 1 
    width=0.05                      #change width to change the shape of the ellipse
    n = len(P)
    dP = P.reshape(1,n,3) - P.reshape(n,1,3)
    D = np.hypot(dP[...,0], dP[...,1])
    D = np.minimum(np.hypot(dP[...,0]+1000, dP[...,1]),D)
    D = np.minimum(np.hypot(dP[...,0], dP[...,1]+1000),D) 
    D = np.minimum(np.hypot(dP[...,0]+1000, dP[...,1]+1000),D)
    # Shifted Distances 
    D = np.hypot(width*D, dP[...,2]+d)
    #W = np.zeros((n,n))
    W=np.where((D-sigma)<0, 1 , 0)#np.where((np.random.uniform(0,1,(n,n)) < np.exp((-np.power(np.maximum(0,D-4*sigma),2))/(2*(sigma/2)**2))), 1, 0)
    s=np.argwhere(W==1)
    for i in range(s.shape[0]):
        if(P[s[i,1],2]>=P[s[i,0],2]):
            W[s[i,0],s[i,1]]=0
    return W



def abstract_layer(in_index, W, t):                                            #unfold the total connection matrix into layer by layer connection matrix. returns the list of matrix.
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
    index=in_index                                                               #This make sure that at each layers neurons are connected to at least n_min neurons deleting the neurons not having enought backward connection. If n_min=1, it doesn't delete qny neurons.
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
    
