"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np
import math
import sklearn
_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    return (np.inner(X1,X2)+1)**_polyDegree



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d = X1.shape
    n2,d = X2.shape
    X1square = (X1**2).dot(np.ones((d,n2)))
    X2square = np.dot((np.ones((n1,d))),(X2**2).T)

    result = -2*X1.dot(X2.T) + X1square + X2square
    result = np.exp((-result/(2*_gaussSigma)))
    
    return result