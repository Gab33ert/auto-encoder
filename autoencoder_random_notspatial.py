#Train auto encoder with spatial neuron gas but handcrafted layers. Then train backpropagation on layers. Compare result for all to all and sparsely spatially connected network. Cannot have more than 3 layers (l=3) modify the function split if needed.
import os
import tqdm
import voronoi
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from random import randint
import copy
from sklearn import preprocessing
import time
from scipy import sparse

data_size=1000 # size of both training and testing dataset
l=3
size=30 #input/output size

def connect(Pl, l, n_input_cells, n_output_cells):
    sigma=340
    d=200
    W=[]
    for j in range(l-1):
        P1=Pl[j]
        P2=Pl[j+1]
        n1 = len(P1)
        n2 = len(P2)
    
        dP = P1.reshape(1,n1,2) - P2.reshape(n2,1,2)
    
            
    
        # Distances
        
        D = np.hypot(dP[...,0]+d, dP[...,1])
        #D = dP[...,1]
    
        w = np.zeros((n2,n1))
        for i in range(n1):
                for j in range(n2):
                    if (np.random.uniform(0,1) < np.exp(-(D[j,i]**2)/(2*sigma**2))):
                        w[j,i]=1
        W.append(w)
    return W

def split(P, n_input_cells):#l number of layer 
    n=len(P)
    layer=[]
    P=P[np.argsort(P[:,0]),:]
    layer.append(P[0:n_input_cells,:][np.argsort(P[0:n_input_cells,:][:,1]),:])
    P=P[n_input_cells:P.shape[0],:]
    layer.append(P[0:P.shape[0]-n_input_cells,:])
    layer.append(P[P.shape[0]-n_input_cells:P.shape[0],:][np.argsort(P[P.shape[0]-n_input_cells:P.shape[0],:][:,1]),:])
    return np.array(layer)


def build(n_cells=1000, n_input_cells = 32, n_output_cells = 32,
          n_input = 3, n_output = 3, sparsity = 0.01, seed=0,l=10):
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
    
    np.random.seed(seed)
    density    = np.ones((1000,1000))
    n=1000
    for i in range(n):
	    ii=i/(n-1)
	    density[:,i]=np.power(((3.85)*ii*(ii-1)+1)*np.ones((1,n)),0.75) #neurone density
    density_P  = density.cumsum(axis=1)
    density_Q  = density_P.cumsum(axis=1)
    filename = "autoencoder-second-degree-b.npy"#aba.npy#autoencoder-second-degree.npy

    if not os.path.exists(filename):
        cells_pos = np.zeros((n_cells,2))
        cells_pos[:,0] = np.random.uniform(0, 1000, n_cells)
        cells_pos[:,1] = np.random.uniform(0, 1000, n_cells)
        for i in tqdm.trange(75):
            _, cells_pos = voronoi.centroids(cells_pos, density, density_P, density_Q)
        np.save(filename, cells_pos)

    cells_pos = np.load(filename)
    cells_pos=split(cells_pos,n_input_cells)
    #X,Y = cells_pos[:,0], cells_pos[:,1]

    
    W=connect(cells_pos, l, n_input_cells, n_output_cells)
    scheme_vis(cells_pos, W)
    return cells_pos/1000, W#, W_in, W_out, bias
    

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    x=sigmoid(x)
    return x*(1-x)

def forward(x, w, b):
    l=len(w)
    X=[]
    for i in range(l):
        X.append(x)
        x=np.asarray(sigmoid(w[i].dot(x)+np.transpose(np.asmatrix(b[i]))))
        #x=np.asarray(sigmoid((sparse.csr_matrix(w[i]).dot(sparse.csr_matrix(x))).todense()))
    return X, x


def backprop(x_in, x_in_t, w, b, alpha, Spatial): #propagate back, train W one step and resturn actual error
    X, x_out=forward(x_in,w,b)
    X_t, x_out_t=forward(x_in_t,w,b)
    l=len(w)-1
    e=[dsigmoid(w[l].dot(X[l]))*(x_out-x_in)]
    for i in range(l-1,-1,-1):
        e.append(dsigmoid(w[i].dot(X[i]))*np.transpose(w[i+1]).dot(e[l-1-i]))
    if Spatial:
        for i in range(l+1):
            for j in range(x_in.shape[1]):
               w[i]-=alpha*(np.outer(e[l-i][:, j], X[i][:, j]))*W[i]
               b[i]-=alpha*e[l-i][:, j]
    else:
        for i in range(l+1):
            for j in range(x_in.shape[1]):
               w[i]-=alpha*(np.outer(e[l-i][:, j], X[i][:, j]))
               b[i]-=alpha*e[l-i][:, j]
    return w, (np.sum((x_out-x_in)**2))/(x_in.shape[1]), np.mean(np.abs(x_out_t-x_in_t))

    
def train(x_in, x_in_t, w, b, iterr, alpha, Spatial):
    error=np.zeros(iterr)
    error_t=np.zeros(iterr)
    g=0
    d=0
    j=0
    for i in range(iterr):
        alpha*=(25)**(1/(-iterr))
        x_batch=x_in[:,d:d+32]
        g+=1
        j+=1
        if j==iterr//20: 
            print(100*i/iterr, "%")
            j=0
        if g==32:
            d+=32
            g=0
        if d>data_size-34:
            d=0
        w, error[i], error_t[i]= backprop(x_batch, x_in_t, w, b, alpha, Spatial)
    return w, error, error_t




def generate_poly(data_size, n, degree):
    data=np.zeros((data_size, n))
    def poly(x, param):
        p=0
        for i in range(len(param)):
            p=p*x+param[i]
        return p
    for i in range(n):
        a=2*np.random.random([degree+1])-1
        data[:,i]=poly(np.linspace(-2, 2, data_size), a)
    return data

def scheme_vis(layer, W):
    m=["^", "o", "v"]
    conteur1=0
    for i in layer:
        plt.scatter(i[:,0], i[:,1], marker=m[conteur1])
        conteur1+=1
    plt.axis("off")
    plt.axis('equal')
    #plt.savefig("fig1.pdf")
    plt.show()
    
    c=['b', 'g', 'b', 'y']
    conteur1=0
    conteur2=0
    fig, ax = plt.subplots()
    for i in range(len(W)):
        conteur1+=np.sum(W[i])
        conteur2+=(W[i].shape[0])*(W[i].shape[1])
        lines=[]
        for n in range(W[i].shape[0]):
            for k in range(W[i].shape[1]):
                if W[i][n,k]==1 and np.random.random()>0.75:
                    lines.append([layer[i][k], layer[i+1][n]])
        seg=LineCollection(lines, colors=c[i-1], zorder=0)
        ax.add_collection(seg)
    conteur1=0
    for i in layer:
        plt.scatter(i[:,0], i[:,1], zorder=1, marker=m[conteur1])
        conteur1+=1
    plt.axis("off")
    ax.axis('equal')
    #plt.savefig("fig2.pdf")
    plt.show()
    print(conteur1, conteur2)
    
        


# Build
 


P, W = build(81, size, size, l=l, n_input=1, n_output=1,sparsity=0.05, seed=2)#build network and layers
w=copy.deepcopy(W)

b=[]
for i in range(len(w)):#initialize layers
    print(w[i].shape)
    b.append(np.random.normal(0, 0.1, w[i].shape[0]))
    w[i]=w[i]*np.random.normal(0, 0.1, w[i].shape)

#x=np.random.random([size])
#data=np.random.random([size, 1])

scaler = preprocessing.MinMaxScaler()
data=generate_poly(size, data_size, 4)
scaled_data=scaler.fit_transform(data)
unscaled_data=scaler.inverse_transform(scaled_data)
data_t=generate_poly(size, data_size, 4)
scaled_data_t=scaler.transform(data_t)


a,bb,e=train(scaled_data, scaled_data_t, w, b,1000, 0.5, True)#train the spatialized network (if Spatial=True and sparse initial weights)
X, x=forward(scaled_data_t,w,b)


fig = plt.figure()#polinoms and reconstructions
for i in range(4):
    ax1 = fig.add_subplot(221+i)
    if i==3 or i==2:ax1.set_xlabel('Input and output neurons')
    #if i==0 or i==2:ax1.set_ylabel('Input and output neurons value')
    if i==3:
        plt.plot(x[:,2+i:3+i], label="Reconstructed polynomials") 
        plt.plot(scaled_data[:,2+i:3+i], label="Test polynomials")
        plt.legend(loc="lower right", fontsize="x-small")
        
    else:
        plt.plot(x[:,2+i:3+i]) 
        plt.plot(scaled_data[:,2+i:3+i])
#plt.savefig("fig4.pdf")
plt.show()


li=[]
for i in range(len(X)):
    a=np.mean(np.abs(X[i]), axis=1)
    print(a.shape[0], np.mean(a))
    li.append(a.shape[0]*np.mean(a))
a=np.mean(np.abs(x), axis=1)
print(a.shape[0], np.mean(a))
li.append(a.shape[0]*np.mean(a))

b=[]
for i in range(len(w)):
    b.append(np.random.normal(0, 0.1, w[i].shape[0]))
    w[i]=np.random.normal(0, 0.1, w[i].shape)
    

c,d,f=train(scaled_data, scaled_data_t, w, b, 1000, 0.5, False)#train the all to all network (if Spatial=False and complete initial weights)



fig = plt.figure()#error on training set
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Mean Absolute Value Error')
ax1.set_xlabel('Iterration')

plt.loglog(bb, label="Spatial")
plt.loglog(d, label="not Spatial")
plt.ylim=((0, 0.55))
#plt.savefig("fig3.png")
plt.show()

fig = plt.figure()#error on test set
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Mean Absolute Value Error')
ax1.set_xlabel('Iterration')
plt.set_ylim=(0, 0.55)
plt.plot(e, label="Spatial")
plt.plot(f, label="All to all")
plt.legend()
#ax1.set_xscale('log')
#plt.savefig("fig3.pdf")
plt.show()

print("error%", 100*e[len(e)-1])

