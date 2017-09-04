# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:54:12 2017

@author: ivan
"""

"""
Vamos a intentar crear una función que nos dé el resultado de Least Squares
en 2D
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

DEGREE = 9
ORDER = DEGREE + 1
N_KNOTS = 10

knots = np.arange(0,N_KNOTS,1)
x = np.random.randint(N_KNOTS, size=N_KNOTS)
y = np.random.randint(N_KNOTS, size=N_KNOTS)

def least_squares(x,y, degree, my_lambda):
    X = np.tile(x,(degree,1)).T
    X = np.cumprod(X, axis=1)
    X = np.hstack((np.ones((N_KNOTS,1)),X))

    A = my_lambda * np.eye(degree + 1) + np.dot(X.transpose(),X)
    B = np.dot(X.transpose(),y[:,np.newaxis])
    
    W = solve(A,B)
    
    return W
    
def compute_curve(x, y, degree, my_lambda):
    W_x = np.ndarray.flatten(least_squares(knots, x, degree, my_lambda))
    W_y = np.ndarray.flatten(least_squares(knots, y, degree, my_lambda))
    
    curve_i = np.arange(0,N_KNOTS-1+0.01,0.01)
    x_i = np.tile(curve_i,(DEGREE,1)).T
    x_i = np.cumprod(x_i, axis=1)
    x_i = np.hstack((np.ones((x_i.shape[0],1)),x_i))
    curve_x = np.sum(W_x * x_i, axis=1)
    
    y_i = np.tile(curve_i,(DEGREE,1)).T
    y_i = np.cumprod(y_i, axis=1)
    y_i = np.hstack((np.ones((y_i.shape[0],1)),y_i))
    curve_y = np.sum(W_y * y_i, axis=1)

    return curve_x, curve_y

MY_LAMBDAS = np.arange(-1e-5,1e-5,1e-6)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x, y, 'o')

for lambdita in MY_LAMBDAS:
    curve_x, curve_y = compute_curve(x, y, DEGREE, lambdita)
    ax.plot(curve_x, curve_y)

plt.plot()
plt.show()
