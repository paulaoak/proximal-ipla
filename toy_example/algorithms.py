import os
import sys

#######################
## TOY EXAMPLE
#######################

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from tqdm import tqdm
import numpy as np

def sig(x):
 return 1/(1 + np.exp(-x))

def gradient_likelihood(x, y): 
    return -(x - y)

lambda_2 = 0.1
def gradient_prior(x, theta): 
    return - lambda_2 * (x - theta)


#####################################
### MOREAU-YOSIDA LANGEVIN ALGORITHMS
#####################################

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


def mypipla(th, X, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Proximal Interacting Particle Unadjusted Langevin algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_likelihood(Xk, data) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma + np.sqrt(2 * h/N) * np.random.normal(0, 1, 1)
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X


def mypgd(th, X, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Proximal Interacting Particle Unadjusted Langevin algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk * (1-h/gamma) + h * gradient_likelihood(Xk, data) + h * proximal_output_particles/gamma + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] * (1-h/gamma) + h * proximal_output_theta/gamma 
        
        X = np.append(X, Xkp1, axis=1) # Store updated cloud.
        th = np.append(th, thkp1)  # Update theta.

    return th, X


def pipgla(th, X, data, proximal_map = proximal_map_laplace_approx, N = 100, K = 4000, gamma = 0.001, h = 0.001, progress_bar=True):
    """
    Run the Proximal Interacting Particle Unadjusted Langevin algorithm for a given proximal mapping.
    """

    for k in (tqdm(range(K), disable=not progress_bar)):

        Xk = X[:, -N:]

        proximal_output_theta_expand, proximal_output_particles = proximal_map(th[k], Xk, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        Xkp1 =  Xk + h * gradient_likelihood(Xk, data) + np.sqrt(2*h) * np.random.normal(0, 1, Xk.shape)
        thkp1 = th[k] + np.sqrt(2*h/N) * np.random.normal(0, 1, 1)

        proximal_output_theta_expand, proximal_output_particles = proximal_map(thkp1, Xkp1, gamma = gamma)  
        
        proximal_output_theta = proximal_output_theta_expand.mean(axis = 1)
        
        X = np.append(X, proximal_output_particles, axis=1) # Store updated cloud.
        th = np.append(th, proximal_output_theta)  # Update theta.

    return th, X