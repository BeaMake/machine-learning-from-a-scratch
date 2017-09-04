# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:18:49 2017

@author: ivan
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

N = 2
N_SAMPLES = 100
MU_0 = np.array([0.6,0.75])
SIGMA_0 = np.array([[0.1, 0.02], [0.15, 0.1]])
MU_1 = np.array([0,0])
SIGMA_1 = np.array([[0.1, 0.2], [0, 0.2]])

sample_0 = np.random.multivariate_normal(MU_0, SIGMA_0, N_SAMPLES)
sample_1 = np.random.multivariate_normal(MU_1, SIGMA_1, N_SAMPLES)

plt.plot(sample_0[:,0], sample_0[:,1], 'o', color='g')
plt.plot(sample_1[:,0], sample_1[:,1], 'o', color='r')

"""
Ya tenemos los datos generados de cada muestra. Ahora 
vamos a crear una función que estime los parámetros de cada
distribución normal multlivariable a partir de los datos de entrada
"""

# Vamos a necesitar esto para calcular SIGMA
def dot_product_transposed(x):
    b = np.tile(x,(2,1))
    a = x[:,np.newaxis]
    return a*b

def calcular_params(sample):
    mu = np.mean(sample, axis=0)
    A = sample - mu
    B = np.empty((N_SAMPLES,2,2))
    for i in range(N_SAMPLES):
        B[i] = dot_product_transposed(A[i])
    sigma = 1/N_SAMPLES * np.sum(B,axis=0)
    return mu, sigma

# La función de la normal multivariable
def mult_normal(x, MU, SIGMA):
    """
    @param x: numpy array
    @return : single value
    """
    cte = (2 * np.pi)**(N/2) * np.abs(np.sqrt(np.linalg.det(SIGMA)))
    A = x - MU
    return (1/cte) * np.exp(-0.5 * np.dot(A,np.dot(A,np.linalg.inv(SIGMA))))


# Para trazar la frontera, que es nada más que la línea en la que intersecan
# las dos superficies, evaluamos ambas, calculamos su resta, y dejamos las 
# que estén muy próximas a cero
x_norm = np.arange(-1,2,0.05)
y_norm = np.arange(-1,2,0.05)
x_norm, y_norm = np.meshgrid(x_norm, y_norm)
z = np.stack((x_norm, y_norm), axis=2)
mu, sigma = calcular_params(sample_0)
z_norm_0 = np.apply_along_axis(mult_normal, 2, z, mu, sigma)
mu, sigma = calcular_params(sample_1)
z_norm_1 = np.apply_along_axis(mult_normal, 2, z, mu, sigma)

diffs = z_norm_0 - z_norm_1
#xss = x_norm[abs(diffs< 1e-32)]
#yss = y_norm[abs(diffs< 1e-32)]

xss = x_norm[z_norm_0 > z_norm_1]
yss = y_norm[z_norm_0 > z_norm_1]
plt.plot(xss, yss, 'p', color='g', alpha=0.3)
xss = x_norm[z_norm_0 < z_norm_1]
yss = y_norm[z_norm_0 < z_norm_1]
plt.plot(xss, yss, 'p', color='r', alpha= 0.25)