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
from collections import OrderedDict

from .core import dtype

def adam(loss, params, learning_rate=0.001, beta1=0.9, beta2=0.999, 
         epsilon=1e-8):

    all_grads = T.grad(loss, params)
    t_prev = theano.shared(np.asarray(0., dtype=dtype))
    updates = OrderedDict()

    t = t_prev + 1
    one = T.constant(1)
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)
    
    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        
        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


