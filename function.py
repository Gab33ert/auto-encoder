#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#functions for rbm propagation, as we sample in [-1,1] instead of [0,1] we had small modifications.
def sigmoid(x):
    return 1/(1+np.exp(-2*x))#(2/(1+np.exp(-x)))-1

def dsigmoid(x):
    f=sigmoid(x)
    return 2*f*(1-f)#-(f+1)*(f-1)/2

