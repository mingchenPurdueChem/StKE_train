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
import os,sys
import numpy as np

import theano
from dmap.draw import draw_contour
from dmap.nn import MLP


#DATA_FILE = './data/aladi-4dim.txt'
#WEIGHT_FILE = './data/weight.out'
DATA_FILE = sys.argv[1]
WEIGHT_FILE = sys.argv[2]
#N = 500*1000
N=8611 # Put in the number of samples
s1=0.3 # standard deviation for density estimator
s2=0.7 # standard deviation for diffusion distance measure.
n_epoches=5000 # Total number of optimization steps ( Stochastic OPT is under development)
plot_every=10000 # plot...not very useful... If your matplotlib does not support, please use a very large number

Y_FILE = './data/y.out'
SEED = None

def read_data(fname, cols, has_id=True):
    data = np.zeros(shape=(N, cols), dtype=theano.config.floatX)    

    with open(fname, 'r') as fobj:
        ID = -1
        for line in fobj:
            line=line.strip()
            if line=='': continue
            if line[0]=='#': continue

            words = line.split()
            if has_id:
                if len(words)!=cols+1: print(line)
                else: assert(len(words)==cols+1)
            else:
                assert(len(words)==cols)

            #try:
            if has_id:
                ID = int(float(words[0]))
                coords = [float(i) for i in words[1:]]
            else:
                ID += 1
                coords = [float(i) for i in words]

            if ID >= N:
                print('ignore: ', line)
                break
            data[ID, :] = np.asarray(coords)
            #except:
            #    print('ignore: ', line)
    return data

def read_weights(fname):
    w = np.zeros(shape=(N,), dtype=theano.config.floatX)
    
    with open(fname, 'r') as fobj:
        ID = 0
        for line in fobj:
            line=line.strip()
            if line=='': continue
            if line[0]=='#': continue

            w[ID] = float(line)
            ID+=1
            if ID >= N: break
        assert(ID == N)
    w *= (1./np.sum(w))
    return w

def subsample(X, y, size, random_state=None):
    shuffle = np.random.permutation(X.shape[0])
    X, y = X[shuffle[0:size]], y[shuffle[0:size]]

    return X, y

def write_y_file(iarr, yarr, warr, fname):
    with open(fname, 'w') as fobj:
        for i in range(yarr.shape[0]):
            x, y = yarr[i, :]
            fobj.write('{:d} {:d}   {:16.8f} {:16.8f}   {:16.8e}'.format(i,
                       iarr[i], x, y, warr[i])+os.linesep)
    return

def main():
    raw_data = read_data(DATA_FILE, cols=10, has_id=True)
    print('sample data: ', raw_data[0])
    print('sample data: ', np.sin(raw_data[0]))
    print('sample data: ', np.cos(raw_data[0]))

    data = np.zeros(shape=(raw_data.shape[0], raw_data.shape[1]*2), dtype=np.float32)
    data[:,0::2] = np.sin(raw_data)
    data[:,1::2] = np.cos(raw_data)
    print(data.shape)
    print(data[0])
    
    weights = read_weights(WEIGHT_FILE)
    print("Coord shape: ", data.shape)
    print("weigth shape: ", weights.shape)    

    indices = np.arange(data.shape[0])
    selected = indices
    #selected = indices[weights>5e-5]
    #selected = np.random.permutation(indices[weights>1e-600])[:9000]
    #selected = np.asarray([i for i, w in zip(indices, weights)\
    #                       if w>2e-5 and w<0.0002])
    print(selected)
    print(selected.shape)
    X = data[selected, :]
    W = weights[selected]
    print("X Shape:", X.shape)
    print("W Range:", W.min(), W.max())

    #G = data[selected, 0:2]
    #draw_contour(G[:,0], G[:,1], W, 'original')

    #shuffle = np.arange(data.shape[0])
    #shuffle = shuffle[:10*1024]
    #X = data[shuffle]
    #W = weights[shuffle]

    # sigma is chosen between 2.0 to 0.1 ...
    mlp = MLP(X.shape[0], X.shape[1], output_dims=2)
    clf = mlp.optimize(x=X, w=W, s1=0.3, s2=0.7, n_epoches=5000, plot_every=1000)
    xl = clf[:, 0]
    yl = clf[:, 1]
    draw_contour(xl, yl, W)
    write_y_file(selected, clf, W, Y_FILE)

    json_string = mlp.to_json()
    with open("out.json", 'w') as fobj:
        fobj.write(json_string)

    with open("hist.txt", 'w') as fobj:
        for a,b,w in zip(xl,yl,W):
            fobj.write("%12.6f   %12.6f   %12.6e"%(a,b,w)+os.linesep)

if __name__=='__main__':
    main()


