# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:40:13 2017

@author: ivan
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

N = 2
N_SAMPLES = 200
MU = np.array([0,0])
SIGMA = np.array([[0.1, 0], [0, 0.2]])

def mult_normal(x):
    """
    @param x: numpy array
    @return : single value
    """
    cte = (2 * np.pi)**(N/2) * np.abs(np.sqrt(np.linalg.det(SIGMA)))
    A = x - MU
    return (1/cte) * np.exp(-0.5 * np.dot(A,np.dot(A,np.linalg.inv(SIGMA))))

def dot_product_transposed(x):
    b = np.tile(x,(2,1))
    a = x[:,np.newaxis]
    return a*b
    
sample = np.random.multivariate_normal(MU, SIGMA, N_SAMPLES)


print "MU"
print MU
print "SIGMA"
print SIGMA
print
print "Estimated MU"
mu = np.mean(sample, axis=0)
print np.mean(sample, axis=0)
print "Estimated SIGMA"
A = sample - mu
B = np.empty((N_SAMPLES,2,2))
for i in range(N_SAMPLES):
    B[i] = dot_product_transposed(A[i])
sigma = 1/N_SAMPLES * np.sum(B,axis=0)
print sigma

# Plotting ------------
x_norm = np.arange(-1.5,1.5,0.1)
y_norm = np.arange(-1.5,1.5,0.1)
x_norm, y_norm = np.meshgrid(x_norm, y_norm)
z = np.stack((x_norm, y_norm), axis=2)
z_norm = np.apply_along_axis(mult_normal, 2, z)


#plt.plot(sample[:,0], sample[:,1], '.')

fig = plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) # Cuando los ax no son iguales, ni
                            # los queremos dividir en una cuadrícula homogénea, hay 
                            # que usar gridspec. Esto también cambia como se añaden los
                            # subplots
                            # fig.add_subplot(1,3,1) --> plt.subplot(gs[0])
ax1 = plt.subplot(gs[0])
ax1.set_xlim([-1.4,1.4])
ax1.set_ylim([-1.4,1.4])
ax1.plot(sample[:,0], sample[:,1],'x')

ax2 = plt.subplot(gs[1], projection='3d')
ax2.set_subplotspec(gs[0:2])
ax2.plot_surface(x_norm, y_norm, z_norm, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
ax2.contour(x_norm, y_norm, z_norm, 3, zdir='z', offset= 0)
ax1.contourf(x_norm, y_norm, z_norm, 4, cmap="autumn_r", alpha=0.3, linewidth=.5)
plt.show()