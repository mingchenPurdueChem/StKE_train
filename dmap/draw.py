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

import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

def draw_contour(xf, yf, wf, fname=None, index=None):
    """
    Parameters
    ----------
    xf : flatten x coord
    yf : flatten y coord
    wf : flatten w coord
    fname : file name w/o extension
    index : file index
    """
    xi = np.linspace(np.min(xf), np.max(xf), 100)
    yi = np.linspace(np.min(yf), np.max(yf), 100)
    zi = griddata(xf, yf, wf, xi, yi, interp='linear')
    plt.contour(xi, yi, zi, 10, linewidth=0.1, colors='k')
    plt.contourf(xi, yi, zi, 10, cmap=plt.cm.rainbow,
                 vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()
    plt.scatter(xf, yf, c='blue', s=1, alpha=0.5, edgecolors='none', zorder=10)
    if fname:
        if index:
            pic_name = "./pic/%s_%04d.png"%(fname, index)
        else:
            pic_name = "./pic/%s.png"%fname
        plt.savefig(pic_name, dpi=300)
        print("saved ", pic_name)
    else:
        plt.show()
    plt.clf()
    return
