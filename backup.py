import numpy as np
import math
import matplotlib.pyplot as plt

#Discretisation
A, B, N = 0, 500, 101
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta

#Parametres du modele
mu = -5
a = 50
sigma2 = 12

#Donnees
observation_indexes = [0, 20, 40, 60, 80, 100]
depth = np.array([0, -4, -12.8, -1, -6.5, 0])

#Indices des composantes correspondant aux observations et aux composantes non observees
unknown_indexes = list(set(discretization_indexes) - set(observation_indexes))

def cov(dist, a, sigma2):
    SIGMA = np.zeros(np.shape(dist))
    for i in range(len(dist)):
        for j in range(len(dist[0])):
            SIGMA[i][j] = sigma2*np.exp(-abs(dist[i][j])/a)
    return(SIGMA)

def distance(discretization):
    t = len(discretization)
    dist = np.zeros((t,t))
    #Construction de la matrice en utilisant sa propriete d'antisymetrie (en comptant les distances algebriquement)
    for i in range(1,t):
        for j in range(i):
            dist[i][j] = discretization[i] - discretization[j]
    dist = dist - dist.T
    return(dist)