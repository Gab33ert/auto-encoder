#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:03:45 2019

@author: gouraud
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.umath_tests import inner1d

import rbm_train as rbmt


def energy(visible, b, c, w):
    e=0
    hidden=rbmt.sample_rbm_forward(visible, c, w)
    e+=-np.sum(b*visible)-np.sum(c*hidden)-np.sum(inner1d(hidden, w.dot(visible)))
    return e/(visible.shape[1]) 
        
 
def gibbs_sampling(visible, b, c, w, n, scaler):
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
    a=visible
    for i in range(n):
        hidden=rbmt.sample_rbm_forward(a, c, w)
        visible=rbmt.sample_grbm_backward(hidden, b, w)
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
    
def gibbs_deep_sampling(visible, b, c, w, n, scaler):
    #plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    #plt.colorbar()
    #plt.show()
    a=visible
    hidden=rbmt.sample_rbm_forward(a, c[0], w[0])
    for i in range(n):
        hidden=rbmt.sample_rbm_forward(hidden, c[i+1], w[i+1])
    for i in range(n):
        hidden=rbmt.sample_rbm_backward(hidden, b[n-i], w[n-i])
    visible=rbmt.sample_grbm_backward(hidden, b[0], w[0])
    plt.imshow(np.transpose(scaler.inverse_transform(np.transpose(visible))).reshape(28,28), vmin=-100, vmax= 400)
    plt.colorbar()
    plt.show()
      
def fake_data(b, c, w, n):
    hidden=np.random.randint(2, size=c[n].shape)
    hidden, visible=rbmt.backandforw(hidden, b[n], c[n], w[n], 50)
    N=n
    while N >1:
        visible=rbmt.sample_rbm_backward(visible, b[N-1], w[N-1])
        N-=1
    plt.imshow(rbmt.sample_grbm_backward(visible, b[N-1], w[N-1]).reshape(28,28))
    

def error(visible, b,c,w):
    return np.sum(np.abs(visible-rbmt.sample_rbm_backward(rbmt.sample_rbm_forward(visible,c,w),b,w)))/(visible.shape[0]*visible.shape[1])

