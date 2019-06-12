#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:02:18 2019

@author: gouraud
"""
import os
import tqdm
import voronoi
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
np.set_printoptions(threshold=np.inf)
from numpy.core.umath_tests import inner1d
from mpl_toolkits.mplot3d import Axes3D as ax3

#TOOLS
def visualize(P, W, in_index, t):                                              #plot the propagation of neurone activation given Position vector P and connection matrix W
    index=in_index
    l=[]
    for i in range(t):
        l.append(len(index))
        plt.scatter(P[:,0],P[:,1], s=3)
        plt.scatter(P[index][:,0],P[index][:,1], s=3)
        axes = plt.gca()
        axes.set_xlim([0,1])
        plt.show()
        x=np.zeros(W.shape[0])
        for i in index:x[i]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index=[idx for idx, v in enumerate(x) if v]
    print(l)
 
def visualize_3d(P, W, in_index, t):
    index=in_index
    l=[]
    for i in range(t):
        l.append(len(index))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(P[index][:,0],P[index][:,1], P[index][:,2])
        ax.scatter(P[:,0],P[:,1], P[:,2], s=1)
        plt.show()
        x=np.zeros(W.shape[0])
        for i in index:x[i]=1
        x=W.dot(x)
        x=(x > 0).astype(int)
        index=[idx for idx, v in enumerate(x) if v]
    print(l)

def visualize_abstract_3d(P, W, index_list, t):                                   #same as usual visualize function but using the abstract layer list
    fig = plt.figure()
    for i in range(t):
        ax=fig.add_subplot(221+i, projection='3d')
        #ax=fig.gca(projection='3d')
        ax.scatter(P[:,0],P[:,1], P[:,2], s=0.2, color="red")
        #ax.scatter(P[index_list[i+1]][:,0],P[index_list[i+1]][:,1], P[index_list[i+1]][:,2], color="green")
        ax.scatter(P[index_list[i]][:,0],P[index_list[i]][:,1], P[index_list[i]][:,2], s=2, color="green")
        ax.set_title(label="layer "+str(i+1))
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    plt.legend()
    #plt.savefig("3dLayers.pdf")
    plt.show()

    
def analyze_topology_back(Wt, depth):                                           #Plot histogram of the number of neurons in layers number depth that has a given % of connection to the input rate.
    n=Wt[depth-1].shape[0]
    l=[]
    for j in range(n):
        out=np.zeros((1,n))
        out[0,j]=1
        for i in range(depth):
            out=out.dot(Wt[depth-1-i])
            out=(out > 0).astype(int)
        l.append(100*np.sum(out)/out.shape[1])
    plt.hist(l,weights=np.ones(len(l)) / len(l))
    plt.title("backward depth "+str(depth))
    #plt.ylim(bottom=0)
    plt.show()

def connection_backward_rate(Wt):                                               #histogram of the number of neurons in a layer that has a given backward connection rate.
    for i in range(len(Wt)):
        plt.hist(np.sum(Wt[i], axis=1),weights=np.ones(Wt[i].shape[0]) / Wt[i].shape[0])#/Wt[i].shape[1]
        plt.title("backward connection rate from layer "+str(i)+" to "+str(i+1))
        plt.show()
        
def connection_forward_rate(Wt):                                               #histogram of the number of neurons in a layer that has a given forward connection rate.
    for i in range(len(Wt)):
        plt.hist(np.sum(Wt[i], axis=0),weights=np.ones(Wt[i].shape[1]) / Wt[i].shape[1])#/Wt[i].shape[0]
        plt.title("forward connection rate from layer "+str(i)+" to "+str(i+1))
        plt.show()
    
def analyze_topology(Wt, depth):
    n=Wt[depth-1].shape[0]
    out=np.ones((n,1))
    for i in range(depth):
        out=np.transpose(out).dot(Wt[depth-1-i])
        out=np.transpose((out > 0).astype(int))
    print(100*np.sum(out)/out.shape[0])
    

def analyze_topology_froward(Wt, depth):                                        #plot histogram of the number of neurons from the input that has a given connection rate to the layers number depth
    l=[]    
    n=Wt[0].shape[1]
    for j in range(n):
        x=np.zeros((n,1))
        x[j,0]=1
        for i in range(depth):
            x=Wt[i].dot(x)
            x=(x > 0).astype(int)
        l.append(100*np.sum(x)/x.shape[0])
    plt.hist(l,weights=np.ones(len(l)) / len(l))
    plt.title("forward depth "+str(depth))
    #plt.ylim(bottom=0)
    plt.show()

def connection_forward_0(Wt):                                               #detect nerons with 0 output connections
    l=[]
    for i in range(len(Wt)):
        l.append(np.mean(np.where(np.sum(Wt[i], axis=0)==0, 1, 0)))
    print(l)
    
def degree_distribution(Wt):                                                     #plot histogram of the distribution of number of connection per neurone in each rbm
    for i in range(len(Wt)):
        plt.hist(np.append(np.sum(Wt[i], axis=1), np.sum(Wt[i], axis=0)), label="degree distribution layer "+str(i+1), log=True)
        plt.xscale("log")
        plt.legend()
        plt.show()