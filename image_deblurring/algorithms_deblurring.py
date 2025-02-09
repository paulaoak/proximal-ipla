import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from prox_tv import tv1_2d

###########################
## AUXILIARY FUNCTIONS
###########################

def generate_blur_matrix(n1, n2, patch_size=10):
    """Generates a blurring matrix H that uniformly averages over a patch_size x patch_size neighborhood."""
    size = n1 * n2
    H = sp.lil_matrix((size, size)) # This is a structure for constructing sparse matrices incrementally.
    pad = patch_size // 2
    
    for i in range(n1):
        for j in range(n2):
            row_idx = i * n2 + j
            weights = []
            indices = []
            
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n1 and 0 <= nj < n2:
                        col_idx = ni * n2 + nj
                        indices.append(col_idx)
                        weights.append(1)
            
            weights = np.array(weights) / np.sum(weights)
            H[row_idx, indices] = weights
    
    return H.tocsr()

def discrete_gradient(n1, n2):
    """Constructs the discrete gradient operator for total variation."""
    size = n1 * n2
    Dx = sp.lil_matrix((size, size))
    Dy = sp.lil_matrix((size, size))
    
    for i in range(n1):
        for j in range(n2):
            idx = i * n2 + j
            if i < n1 - 1:
                Dx[idx, idx] = -1
                Dx[idx, idx + n2] = 1
            if j < n2 - 1:
                Dy[idx, idx] = -1
                Dy[idx, idx + 1] = 1
    
    return Dx.tocsr(), Dy.tocsr()

def total_variation(x, Dx, Dy):
    """Computes the total variation prior."""
    x_flat = x.reshape(-1, x.shape[2])
    grad_x = Dx @ x_flat
    grad_y = Dy @ x_flat
    return np.sum(np.sqrt(grad_x**2 + grad_y**2), axis=0)


###########################
## ALGORITHMS
###########################

def pipgla(theta, w_init, H, y, sigma, lambdaaa, Dx, Dy, original, K=1000, h=0.01, method_tv = "dr"):
    """Samples from the posterior using PIPGLA with TV proximal operator."""
    
    w = w_init.copy()
    Dw = w[:, :, 0].size  # Dimension of particles.
    N = w.shape[-1] # Number of particles. 

    nmse_values = np.zeros(K)
    theta_values = np.zeros(K) 
    y_flat = y.flatten()
    y_flat = y_flat[:, None]

    for k in tqdm(range(K)):
        # Parameter updates 
        theta = theta - h * np.exp(theta) * np.mean(total_variation(w, Dx, Dy))/(Dw)  + np.sqrt(2*h/N)*np.random.normal(0, 1, 1)
        theta_values[k] = theta

        # Particle updates
        w_flat = w.reshape(-1, w.shape[2])
        grad_w = (-1 / sigma**2) * (H.T @ (H @ w_flat - y_flat))
        w = w + h * grad_w.reshape(w.shape) + np.sqrt(2 * h) * np.random.normal(0, 1, w.shape)
        w = tv1_2d(w, np.exp(theta) * lambdaaa, method = method_tv)  # Apply TV proximal operator

        # Compute MSE
        original_reshape = np.tile(original, (1, 1, N))
        err_aux = np.sum((w - original_reshape)**2, axis=[0,1]) / np.sum((original)**2)
        nmse_values[k] = np.mean(err_aux)

    return w, nmse_values, theta_values


def myipla(theta, w_init, H, y, sigma, lambdaaa, Dx, Dy, original, K=1000, h=0.01, method_tv = "dr"):
    """Samples from the posterior using MYIPLA with TV proximal operator."""
    
    w = w_init.copy()
    Dw = w[:, :, 0].size  # Dimension of particles.
    N = w.shape[-1] # Number of particles. 

    nmse_values = np.zeros(K)
    theta_values = np.zeros(K) 
    y_flat = y.flatten()
    y_flat = y_flat[:, None]

    for k in tqdm(range(K)):
        # Parameter updates 
        theta = theta - h * np.exp(theta) * np.mean(total_variation(w, Dx, Dy))/(Dw)  + np.sqrt(2*h/N)*np.random.normal(0, 1, 1)
        theta_values[k] = theta

        # Particle updates
        w_flat = w.reshape(-1, w.shape[2])
        grad_w = (-1 / sigma**2) * (H.T @ (H @ w_flat - y_flat))
        prox_term_w = tv1_2d(w, np.exp(theta) * lambdaaa, method = method_tv)
        w = w * (1-h/lambdaaa) + h * grad_w.reshape(w.shape) + h/lambdaaa * prox_term_w + np.sqrt(2 * h) * np.random.normal(0, 1, w.shape)

        # Compute MSE
        original_reshape = np.tile(original, (1, 1, N))
        err_aux = np.sum((w - original_reshape)**2, axis=[0,1]) / np.sum((original)**2)
        nmse_values[k] = np.mean(err_aux)

    return w, nmse_values, theta_values


def mypgd(theta, w_init, H, y, sigma, lambdaaa, Dx, Dy, original, K=1000, h=0.01, method_tv = "dr"):
    """Samples from the posterior using MYPGD with TV proximal operator."""
    
    w = w_init.copy()
    Dw = w[:, :, 0].size  # Dimension of particles.
    N = w.shape[-1] # Number of particles. 

    nmse_values = np.zeros(K)
    theta_values = np.zeros(K) 
    y_flat = y.flatten()
    y_flat = y_flat[:, None]

    for k in tqdm(range(K)):
        # Parameter updates 
        theta = theta - h * np.exp(theta) * np.mean(total_variation(w, Dx, Dy))/(Dw)  
        theta_values[k] = theta

        # Particle updates
        w_flat = w.reshape(-1, w.shape[2])
        grad_w = (-1 / sigma**2) * (H.T @ (H @ w_flat - y_flat))
        prox_term_w = tv1_2d(w, np.exp(theta) * lambdaaa, method = method_tv)
        w = w * (1-h/lambdaaa) + h * grad_w.reshape(w.shape) + h/lambdaaa * prox_term_w + np.sqrt(2 * h) * np.random.normal(0, 1, w.shape)

        # Compute MSE
        original_reshape = np.tile(original, (1, 1, N))
        err_aux = np.sum((w - original_reshape)**2, axis=[0,1]) / np.sum((original)**2)
        nmse_values[k] = np.mean(err_aux)

    return w, nmse_values, theta_values
