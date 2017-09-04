# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:08:18 2017

@author: ivan
"""

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def peaks(x,y):
    return x * numpy.sin(y)

fig = plt.figure()
ax = fig.gca(projection='3d')
X = Y= numpy.arange(-3, 3, 0.1).tolist()
X, Y = numpy.meshgrid(X, Y)

Z = []
for i in range(len(X)):
    Z.append(peaks(X[i],Y[i]))

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-8)
cset = ax.contour(X, Y, Z, zdir='x', offset=-8)
cset = ax.contour(X, Y, Z, zdir='y', offset=8)

ax.set_xlabel('X')
ax.set_xlim(-8, 8)
ax.set_ylabel('Y')
ax.set_ylim(-8, 8)
ax.set_zlabel('Z')
ax.set_zlim(-8, 8)

plt.show()