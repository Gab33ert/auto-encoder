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

    
def analyze_topology_back(Wt, depth):
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

def connection_backward_rate(Wt):
    for i in range(len(Wt)):
        plt.hist(np.sum(Wt[i], axis=1)/Wt[i].shape[1],weights=np.ones(Wt[i].shape[0]) / Wt[i].shape[0])
        plt.title("connection rate from layer "+str(i)+" to "+str(i+1))
        plt.show()
    
def analyze_topology(Wt, depth):
    n=Wt[depth-1].shape[0]
    out=np.ones((n,1))
    for i in range(depth):
        out=np.transpose(out).dot(Wt[depth-1-i])
        out=np.transpose((out > 0).astype(int))
    print(100*np.sum(out)/out.shape[0])
    

def analyze_topology_froward(Wt, depth):
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
