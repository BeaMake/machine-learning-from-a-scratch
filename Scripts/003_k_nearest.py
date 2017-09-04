# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:54:06 2017

@author: ivan
"""

from __future__ import division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.patches   import Circle


''' Genera los datos de entrada. Los parámetros mu y sigma no varían mucho '''

def sample(N_DOTS, N_TEAMS):
    dots = np.empty((N_DOTS*N_TEAMS,2))
    SIGMA = np.array([[0.1, 0], [0, 0.2]])
    for i in range(N_TEAMS):
        MU = np.array([i,i])
        dots[i*N_DOTS:(i+1)*N_DOTS] = np.random.multivariate_normal(MU, SIGMA, N_DOTS)
    return dots, np.repeat(range(N_TEAMS), N_DOTS)


''' Este clasificador no se actualiza con los nuevos datos. 
Es facilmente modificable para que sí lo haga '''

class K_nearest:
    def __init__(self, fig, ax, K, dots, teams):
        self.fig = fig
        self.ax = ax
        self.K = K
        self.dots = dots
        self.teams = teams
        self.colors = {0:'silver', 1:'b', 2:'firebrick', 3:'olive', 4:'k', 5:'y'}
        self.circle_list = []
        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
#        self.cid_move = fig.canvas.mpl_connect('motion_notify_event', self.on_move)
#        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.plot_sample()
        
    def K_nearest_points(self, point):
        ''' Calculamos la distancia de 'point' a cada uno de los puntos de la muestra
         Después devolvemos su índice '''
        distances = np.sum((self.dots - point)**2, axis = 1)
        return distances.argsort()[:self.K]
        
    def choose_team(self, point):
        ''' Elige el equipo del punto en función de los K más cercanos '''
        K_nearest = self.K_nearest_points(point)
#        print K_nearest
        K_nearest_teams = np.empty(self.K, dtype = int)
#        print K_nearest_teams
        for i in range(K_nearest.size):
            K_nearest_teams[i] = self.teams[K_nearest[i]]
#        print K_nearest_teams
        count = np.bincount(K_nearest_teams)
        return K_nearest_teams[np.argmax(count)]
    
    def plot_sample(self):
        for i in range(self.teams.size):
            c = Circle((self.dots[i,0], self.dots[i,1]), radius=0.06, color = self.colors[self.teams[i]])
            self.circle_list.append(self.ax.add_patch(c))
            
    def on_press(self, event):
        if event.inaxes != self.ax.axes: return
        if (event.button == 1):
            team = self.choose_team(np.array([event.xdata, event.ydata]))
            c = Circle((event.xdata, event.ydata), radius=0.06, color = self.colors[team])
            self.circle_list.append(self.ax.add_patch(c))
            
if __name__ == '__main__':
    N_DOTS = 20
    N_TEAMS = 4

    fig = plt.figure(num=None, figsize=(8, 7), \
            dpi=80, facecolor='palegoldenrod', edgecolor='k')
    ax = fig.add_subplot(1,1,1, aspect=1, axisbg = 'bisque')
    ax.set_xlim(-2,N_TEAMS + 1)
    ax.set_ylim(-2,N_TEAMS + 1)
    ax.grid(color='coral', linestyle='--', linewidth=0.5)
    
    X, t = sample(N_DOTS, N_TEAMS)
    
    classifier = K_nearest(fig, ax, 15, X, t)
    
    plt.show()
        