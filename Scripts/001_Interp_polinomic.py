# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:59:59 2017

@author: ivan
"""

"""
 Vamos a hacer un pequeño ejemplo de interpolación lineal
 Con una función f:|R2 -> |R. Usaremos el método de mínimos
 cuadrados para hallar los mejores valores para w
"""
 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
from time import time
from mpl_toolkits.mplot3d import Axes3D

N_PARTS = 50
DIM = 2
ORDER = 3 # Interpolación con polinomios cúbicos

def f(x):
    """ Vamos a usar la función x+y 
    pero vamos a introducir una perturbación"""
    eps = np.random.uniform(-1,1,N_PARTS)
    return np.sum(x, axis=1) + eps

# Input data 
X = np.random.uniform(-1,1, size=(N_PARTS,DIM))
X = np.tile(X,ORDER) # Extendemos la matriz para incluir las potencias
# Ahora un atajillo para hacer las columnas de las potencias rápidamente :
X1 = np.ones(X.shape)
for i in range(1,ORDER):
    X2 = np.ones(X.shape) #Creamos una matriz con la forma de X
    X2[:,2*i:] = X[:,2*i:]
    X1 = X1 * X2
X = X * X1

Y = f(X)
# Vamos a usar la forma ampliada, añadiendo una columna de 1s
# para incluir el término independiente en la matriz
X = np.hstack((np.ones((N_PARTS,1)),X))
#Y = np.concatenate((np.ones(1),Y))

# Building the matrix needed for least squares
A = np.dot(X.transpose(),X)
B = np.dot(X.transpose(),Y[:,np.newaxis])

# Vamos a hacerlo multiplicando por la inversa y resolviendo el sistema
# 1. Inversa
#   w = (X.t*X)^(-1).t * X.t * y = B^(-1) * A
t_start = time()
W1 = np.dot(inv(A),B)
t_end = time()
print "Time to compute w using inv:", t_end - t_start
#print W1

# 2. Resolviendo el sistema Ax = B
t_start = time()
W2 = solve(A,B)
t_end = time()
print "Time to compute w using solve:", t_end - t_start
#print W2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,2], Y)

x_plane = np.arange(-1,1,0.1)
y_plane = np.arange(-1,1,0.1)
x_plane, y_plane = np.meshgrid(x_plane, y_plane)
z_plane = W1[0] + W1[1] * x_plane + W1[2] * y_plane + \
                  W1[3] * x_plane**2 + W1[4] * y_plane**2 + \
                  W1[5] * x_plane**3 + W1[6] * y_plane**3

ax.plot_wireframe(x_plane, y_plane, z_plane)

plt.show()