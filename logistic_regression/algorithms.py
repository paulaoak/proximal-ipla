import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tqdm import tqdm
import numpy as np


##############################################################
## UTILS
##############################################################

def sig(x):
 return 1/(1 + np.exp(-x))

def gradient_proximal_logistic_reg(x, l, f): 
    s = 1/(1+np.exp(- np.matmul(f, x)))
    
    return np.matmul((l-s).transpose(), f).transpose()


##############################################################
## PROXIMAL PARTICLE ALGORITHMS AND PROXIMAL MAPS
##############################################################

def pip_ula(proximal_map, th, X, design_matrix, data, N = 100, K = 4000, h = 0.001, progress_bar=True):
    """
    Run the Proximal Interacting Particle Unadjusted Langevin algorithm for a given proximal mapping.
    """
    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, data, design_matrix, gamma = h/2)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 = proximal_output_particles + np.sqrt(h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = proximal_output_theta + np.sqrt(h/N) * np.random.normal(0, 1, 1) 
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X

def prox_pgd(proximal_map, th, X, design_matrix, data, N = 100, K = 4000, h = 0.001, progress_bar=True):
    """
    Run the Proximal Particle Gradient Descent algorithm for a given proximal mapping.
    """
    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, data, design_matrix, gamma = h/2)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 = proximal_output_particles + np.sqrt(h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = proximal_output_theta  
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X

def proximal_map_laplace_approx_total(theta, particles, data, design_matrix, gamma):
    """
    Compute the proximal mapping.
    """

    grad = gradient_proximal_logistic_reg(particles, data, design_matrix)

    input_proximal_x = particles + 2 * gamma * grad

    input_proximal_theta = theta
    
    x_prox = input_proximal_theta + (input_proximal_x - np.sign(input_proximal_x - input_proximal_theta) * gamma - input_proximal_theta) * (np.abs(input_proximal_x-input_proximal_theta) >= gamma)
    theta_prox = input_proximal_theta + np.sign(x_prox - input_proximal_theta).sum(axis = 0) * gamma
    
    proximal_output_x =  x_prox
    proximal_output_theta = theta_prox 
    
    return np.expand_dims(proximal_output_theta, axis=0), proximal_output_x


def proximal_map_laplace_iteration_total(theta, particles, data, design_matrix, gamma):
    """
    Compute the proximal mapping.
    """

    grad = gradient_proximal_logistic_reg(particles, data, design_matrix)

    input_proximal_x = particles + 2 * gamma * grad

    input_proximal_theta = theta

    # Initialize input for the fixed point iteration method.
    x_prox = input_proximal_x 
    theta_prox = input_proximal_theta
    for _ in range(40):
        x_prox = input_proximal_x - np.sign(x_prox - theta_prox) * gamma
        theta_prox = input_proximal_theta + np.sign(x_prox - theta_prox).sum(axis = 0) * gamma

    return np.expand_dims(theta_prox, axis=0), x_prox


def proximal_map_uniform_new(theta, particles, data, design_matrix, gamma):
    """
    Compute the proximal mapping.
    """

    grad = gradient_proximal_logistic_reg(particles, data, design_matrix)

    input_proximal_x = particles + 2 * gamma * grad
    if theta**2 > 4* gamma * 50:
        theta_prop = (theta + np.sqrt(theta**2 - 4* gamma * 50))/2
        theta_proposal = np.array([theta_prop])
    else:
        theta_proposal = np.max(np.abs(input_proximal_x), axis = 0)

    theta_prox = theta_proposal

    x_prox = np.sign(input_proximal_x) * (np.abs(theta_proposal)* (np.abs(input_proximal_x)>=np.abs(theta_proposal))+\
                                          np.abs(input_proximal_x)* (np.abs(theta_proposal)>np.abs(input_proximal_x)))
    
    return np.expand_dims(theta_prox, axis=0), x_prox

##############################################################
### MOREAU-YOSIDA LANGEVIN ALGORITHMS AND PROXIMAL MAPS
##############################################################

def proximal_map_laplace_approx(theta, particles, gamma):
    """
    Compute the proximal mapping approximately for a Laplace prior.
    """

    input_proximal_x = particles 

    input_proximal_theta = theta
    
    x_prox = input_proximal_theta + (input_proximal_x - np.sign(input_proximal_x - input_proximal_theta) * gamma - input_proximal_theta) * (np.abs(input_proximal_x-input_proximal_theta) >= gamma)
    theta_prox = input_proximal_theta + np.sign(x_prox - input_proximal_theta).sum(axis = 0) * gamma
    
    proximal_output_x =  x_prox
    proximal_output_theta = theta_prox 
    
    return np.expand_dims(proximal_output_theta, axis=0), proximal_output_x


def proximal_map_laplace_iterative(theta, particles, gamma):
    """
    Compute the proximal mapping iteratively for a Laplace prior.
    """

    input_proximal_x = particles 

    input_proximal_theta = theta

    # Initialize input for the fixed point iteration method.
    x_prox = input_proximal_x 
    theta_prox = input_proximal_theta
    for _ in range(40):
        x_prox = input_proximal_x - np.sign(x_prox - theta_prox) * gamma
        theta_prox = input_proximal_theta + np.sign(x_prox - theta_prox).sum(axis = 0) * gamma

    return np.expand_dims(theta_prox, axis=0), x_prox


def proximal_map_uniform_my_new(theta, particles, gamma):
    """
    Compute the proximal mapping.
    """
    input_proximal_x = particles
    if theta**2 > 4* gamma * 50:
        theta_prop = (theta + np.sqrt(theta**2 - 4* gamma * 50))/2
        theta_proposal = np.array([theta_prop])
    else:
        theta_proposal = np.max(np.abs(input_proximal_x), axis = 0)

    theta_prox = theta_proposal

    x_prox = np.sign(input_proximal_x) * (np.abs(theta_proposal)* (np.abs(input_proximal_x)>=np.abs(theta_proposal))+np.abs(input_proximal_x)* (np.abs(theta_proposal)>np.abs(input_proximal_x)))
    
    return np.expand_dims(theta_prox, axis=0), x_prox


def mypipla(th, X, design_matrix, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Moreau-Yosida Interacting Particle Langevin Algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_proximal_logistic_reg(Xk, data, design_matrix) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma + np.sqrt(2 * h/N) * np.random.normal(0, 1, 1)
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X


def mypgd(th, X, design_matrix, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Moreau-Yosida Particle Gradient Descent for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_proximal_logistic_reg(Xk, data, design_matrix) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma 
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X



##############################################################
### PROXIMAL GRADIENT LANGEVIN ALGORITHMS
##############################################################

def pipgla(th, X, design_matrix, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Proximal Interacting Particle Unadjusted Langevin algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]
        
        Xkp1 =  Xk + h * gradient_proximal_logistic_reg(Xk, data, design_matrix) + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] + np.sqrt(2*h/N) * np.random.normal(0, 1, 1)

        proximal_output_theta_expand, proximal_output_particles = proximal_map(thkp1, Xkp1, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        X = np.append(X, proximal_output_particles, axis=1) # Store updated cloud.
        th = np.append(th, proximal_output_theta)  # Update theta.

    return th, X
