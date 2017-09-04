# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:54:31 2017

@author: ivan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MU = 0
SIGMA2 = 0.1

N_SAMPLES = 1000

def normal(x):
    return 1/(np.sqrt(2 * np.pi * SIGMA2)) * np.exp(- (x - MU**2)**2/(2 * SIGMA2))

sample = np.random.normal(MU,np.sqrt(SIGMA2), N_SAMPLES)

count, partition = np.histogram(sample)


x_normal = np.arange(-1.5,1.5,0.01)
y_normal = normal(x_normal)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel("Values for distribution")
ax1.set_xlabel(str(N_SAMPLES) + " points generated for this experiment")
ax1.plot(x_normal, y_normal, c='r')

ax2 = ax1.twinx() # Creamos otro ax, que estará en ax1 usando otro eje
ax2.set_ylabel("Count of samples")
ax2.hist(sample, bins='auto', alpha=0.5)

# Ahora vamos a intentar estimar la distribución
MU = np.mean(sample)
SIGMA2 = np.std(sample)**2

y_normal = normal(x_normal)
ax1.plot(x_normal, y_normal, '--', color='g')

plt.show()