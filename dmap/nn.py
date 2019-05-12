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

import json
from .core import *
from .draw import draw_contour
from .updates import adam

class MLP(object):
    def __init__(self, N, num_in, output_dims, lr=0.001,alpha=0.0):
        self.x_t = T.fmatrix('x')
        self.w_t = T.fvector('w')
        self.sigma1_t = T.fscalar('s1')
        self.sigma2_t = T.fscalar('s2')
        l1, (W1, b1) = dense_layer(self.x_t, (num_in, 64), "layer1", b_v=0.1,
                                   nonlinearity=rectify)
        l2, (W2, b2) = dense_layer(l1, (64, 64), "layer2", b_v=0.1,
                                   nonlinearity=rectify)
        l3, (W3, b3) = dense_layer(l2, (64, output_dims), "layer3", b_v=0.1,
                                   nonlinearity=identity)

        cost_t, cost_ref_t = cost_var(self.x_t, l3, self.w_t, 
                                      self.sigma1_t, self.sigma2_t)

        L2 = T.sum(W1**2)+T.sum(W2**2)+T.sum(W3**2)
        #cost_reg = cost_t+alpha*L2
        #updates = adam(cost_reg, [W1, W2, W3, b1, b2, b3], learning_rate=lr)

        updates = adam(cost_t, [W1, W2, W3, b1, b2, b3], learning_rate=lr)
        self.update_func = theano.function(\
                        [self.x_t, self.w_t, self.sigma1_t,
                         self.sigma2_t], [cost_t, l3],
                        updates=updates)

        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def to_json(self):
        def layer_obj(w_v, b_v, act):
            jobj = { }
            jobj['w_dim0'] = w_v.shape[0]
            jobj['w_dim1'] = w_v.shape[1]
            jobj['weights'] = w_v.flatten().tolist()
            jobj['b_dim0'] = b_v.shape[0]
            jobj['biases'] = b_v.flatten().tolist()
            jobj['activation'] = act
            return jobj

        jobj = { 'version': 1000,
                 'num_layers': 3 }
        w1_v = self.W1.get_value()
        w2_v = self.W2.get_value()
        w3_v = self.W3.get_value()
        b1_v = self.b1.get_value()
        b2_v = self.b2.get_value()
        b3_v = self.b3.get_value()

        jlayers = [ layer_obj(w1_v, b1_v, 'RECTIFY'),
                    layer_obj(w2_v, b2_v, 'RECTIFY'),
                    layer_obj(w3_v, b3_v, 'IDENTITY') ]
        jobj['layers'] = jlayers
        return json.dumps(jobj)

    def optimize(self, x, w, s1, s2, n_epoches, plot_every=-1):
        
        for epoch in range(n_epoches):
            c, y = self.update_func(x, w, s1, s2)
            print('[epoch {0}] cost: {1:.6f}, expected {2:.6f}'.format(
                  epoch, float(c), 0.))

            if plot_every > 0 and (epoch + 1) % plot_every == 0 :
                draw_contour(y[:,0], y[:,1], w, fname="DMAP", index=epoch)
        return y

if __name__=='__main__':
   pass 
