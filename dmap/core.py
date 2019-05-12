###############################################################################
#    Copyright 2018 Jing Zhang and Ming Chen                                  #   
#                                                                             #
#    This file is part of StKE_train, version 1.0                             #
#                                                                             #
#    StKE_train is free software: you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    StKE_train is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with StKE_train.  If not, see <https://www.gnu.org/licenses/>.     #
###############################################################################

import theano
import theano.tensor as T
import numpy as np

epsilon=1e-16
dtype=theano.config.floatX

def euclidean_dist(X):
    """
    Compute squared distance map.

    Parameters
    ----------
    X : matrix
        A tensor matrix with dimension (number of samples, number of features)

    Return
    ------
    A tensor matrix for squared distance map.
    """
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - \
           2*X.dot(X.T)

def euclidean_dist2(X, X_new):
    """
    compute cross distance.

    returns
    -------
        square distance map with dim(X_new, X).
    """
    N0 = X_new.shape[0]
    N1 = X.shape[0]

    ss0 = (X_new ** 2).sum(axis=1)
    ss1 = (X ** 2).sum(axis=1)
    return ss0.reshape((N0, 1)) + ss1.reshape((1, N1)) - \
           2*X_new.dot(X.T)

def density_est(X_t, W_t, sigma, X_s):
    """
    Density estimator.

    Parameters
    ----------
    X_t : A tensor matrix
        sample features.
    W_t : A tensor vector
        sample weights.
    sigma : A tensor scalar
        standard deviation for gaussian kernel.
    X_s : A tensor matrix
        input sample features.

    Return
    ------
    density estimation at input sample points.
    """
    N0 = X_s.shape[0]
    N1 = X_t.shape[0]

    sqdist = euclidean_dist2(X_t, X_s)
    #k = T.exp(-sqdist/(2*sigma**2)) * (1./np.sqrt(2*sigma**2*np.pi)) * W_t.reshape((1, N1))
    # Prefactor of Gaussians will cancel
    k = T.exp(-sqdist/(2*sigma**2))* W_t.reshape((1, N1))
    return k.sum(axis=1) * (1./W_t.sum())

def transition_matrix(d_s, X_s, sigma):
    """
    Compute diffusion matrix.
    
    Parameters
    ----------
    d_s : A tensor vector
        density estimation at input points.
    X_s : A tensor matrix
        input sample features.
    sigma : A tensor scalar
        standard deviation for diffusion distance measure.

    Return
    ------
    diffusion matrix, row major.
    """
    N = d_s.shape[0]
    sqdist = euclidean_dist(X_s)
    dr_s = T.sqrt(d_s).reshape((1, N))
    #L = T.exp(-sqdist/(2*sigma**2)) * (1./np.sqrt(2*sigma**2*np.pi)) * (1./dr_s)
    #L = T.exp(-sqdist/(2*sigma**2)) * (1./np.sqrt(2*sigma**2*np.pi)) * dr_s
    # Prefactor of Gaussians will cancel
    L = T.exp(-sqdist/(2*sigma**2)) * dr_s

    row_sum = T.sum(L, axis=1).reshape((N,1))
    return L/row_sum

def transition_student_matrix(d_s, X_s):
    N = d_s.shape[0]
    sqdist = euclidean_dist(X_s)
    dr_s = T.sqrt(d_s).reshape((1, N))
    L = 1./np.pi * 1./(1. + sqdist) * (1./dr_s)  # student-t

    row_sum = T.sum(L, axis=1).reshape((N,1))
    return L/row_sum

def cost_var(X_t, Y_t, W_t, sigma_den, sigma_trans):
    """
    KL divergence estimator.

    Parameters
    ----------
    X_t : A tensor matrix
        high-dimension input features.
    Y_t : A tensor matrix
        low-dimension input features.
    W_t : A tensor vecotr
        sample weights.
    sigma_den : A tensor scalar
        standard deviation for density estimator.
    sigma_trans : A tensor scalar
        standard deviation for diffusion distance measure.

    Return
    ------
    cost value based on KL divergence.
    """

    DX = density_est(X_t, W_t, sigma_den, X_t)
    PX = transition_matrix(DX, X_t, sigma_trans)
    DY = density_est(Y_t, W_t, sigma_den, Y_t)
    PY = transition_matrix(DY, Y_t, sigma_trans)

    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    cost = T.sum(PX * T.log(PXc / PYc))/X_t.shape[0]
    cost_ref = T.sum(PX * T.log(PXc / PXc))/X_t.shape[0]
    return cost, cost_ref

def init_glorot(shape):
    """
    Weight initializor.
    """
    if len(shape) < 2:
        raise RuntimeError("shape should be at least 2D")
    
    n1, n2 = shape[:2]
    field_size = np.prod(shape[2:])
    std = np.sqrt(2.0 / ((n1 + n2) * field_size))
    return np.asarray(np.random.normal(size=shape, loc=0., scale=std),
                      dtype=dtype)

def init_constant(shape, value=0.):
    """
    Constant initializor.
    """
    return np.ones(shape=shape, dtype=dtype) * value

def tanh(x):
    """
    tanh function.
    """
    return theano.tensor.tanh(x)

def rectify(x):
    """
    Rectify Linear Unit.
    """
    return theano.tensor.nnet.relu(x)

def identity(x):
    """
    identity function.
    """
    return x

def dense_layer(X_t, w_shape, name, b_v=None, nonlinearity=tanh):
    """
    Dense Layer.
    
    Parameters
    ----------
    X_t : A tensor matrix.
        input tensor matrix.
    w_shape : 2 dim list.
        weight shape, (number of input units, number of output units).
    name : string.
        layer name, which is passed to tensor.
    b_v : float.
        bias initital value.
    nonlinearity : method
        non-linear activation function.

    Return
    ------
    A list of (output tensor, (weight tensor, bias tensor)).
    """
    if len(w_shape) != 2:
        raise RuntimeError("w_shape should be 2 dim")

    W_value = init_glorot(w_shape)
    W_s = theano.shared(W_value, name="%s:W"%name)

    if b_v is not None:
        b_value = init_constant(w_shape[1], b_v)
        b_s = theano.shared(b_value, name="%s:b"%name)
    else:
        b_s = None

    act = T.dot(X_t, W_s)
    if b_s is not None:
        act = act + b_s
    return nonlinearity(act), [W_s, b_s]
    

