import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch 
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##############################################################
## MOREAU-YOSIDA INTERACTING PARTICLE LANGEVIN ALGORITHM
##############################################################

def pip_myula_matrix_sparse(Y_s, mask_s, h, K, a, Theta, X, gamma, dim_bias, true_matrix = None, theta_true = None):
    # Extract dimensions of latent variables:
    Dx = X[:, :, :, 0].size  # Dimension of X.
    N = X.shape[-1] # Number of particles.
    DY = dim_bias
    Thetas = []
    Thetas.append(Theta)

    # Initialize arrays storing performance metrics as a function of k:
    error = np.zeros(K)  # Test error.
    log_density_vec = np.zeros(K)  # Log density.
    error_missing = np.zeros(K)  # Test error on missing entries.
    error_theta = np.zeros(K)  # Test error of basis matrix Theta.

    for k in (tqdm(range(K))): 
        # Evaluate metrics for current particle cloud:
        if (true_matrix is not None) & (theta_true is not None):

            error[k] = mean_square_error(true_matrix, X)

            theta_recover = Thetas[k]
            
            recover_Y = jnp.tensordot(theta_recover, X, axes=([1], [0]))
            true_Y = jnp.tensordot(theta_true, true_matrix, axes=([1], [0]))

            error_missing[k] = mean_square_error_missing(true_Y, recover_Y, mask_s)

            log_density_vec[k]= jnp.mean(log_density(Y_s, recover_Y, mask_s, a[k]))

            error_theta[k] = jnp.sum((theta_true - theta_recover)**2)/jnp.sum(theta_true**2)

        # Temporarily store current particle cloud:
        Xk = X  # Matrix estimated entries.
        Theta_k = Thetas[k]  # Basis matrix.

        # Update parameter estimates using heuristic 

        a = np.append(a, a[k]* (1-h/gamma) + h*ave_proximal_param_reg(Xk, a[k], gamma)/(DY*gamma)
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Alpha.
        
        Thetas.append(Theta_k - h*ave_grad_param_basis(X, mask_s, Y_s, Theta_k)/DY
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, Theta_k.shape))  # Theta basis matrix.

        # Update particle cloud:
        X = (X * (1-h/gamma) - h * wgrad(X, mask_s, Y_s, Theta_k) + h * approx_proximal_particle(Xk, a[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, X.shape)) 

    return a, Thetas, X, error, error_missing, log_density_vec, error_theta

##############################
# Auxiliary functions.
##############################

# Functions for the log density.

def _log_likelihood(Y, B, mask):
    # Log of a Gaussian likelihood, with mean Y and variance 0.01, evaluated at B.
    v = Y - jnp.where(mask, B, 0) 
    return -jnp.sum(v**2)/2

def _log_prior(B, lsig):
    singular_values = jnp.linalg.svd(B, compute_uv=False)
    tr_norm = jnp.sum(jnp.abs(singular_values))
    theta = jnp.exp(-lsig)
    return -theta * tr_norm

def _log_density(Y, B, mask, lsig):
    #print(Y.shape, B.shape, mask.shape)
    # Log of model density, vectorized over particles.
    out = _log_prior(B, lsig) 
    return out + _log_likelihood(Y, B, mask)

@jax.jit
def _log_density_particles(Y, B, mask, lsig):
    """Log-density."""
    #print(Y.shape, B.shape, mask.shape)
    density = jax.vmap(_log_density, in_axes=(2, 2, 2, None)) 
    density_vec = density(Y, B, mask, lsig) 
    return jnp.sum(density_vec)

@jax.jit
def log_density(Y, B, mask, lsig):
    """Log-density."""
    #print(Y.shape, B.shape, mask.shape)
    density = jax.vmap(_log_density_particles, in_axes=(None, 3, None, None)) 
    density(Y, B, mask, lsig) 
    return density(Y, B, mask, lsig) 

# Functions for the gradients or proximal mappings of the log-density.'  
# Gamma is the smoothing parameter.

###
# Proximal mapping for the regularization (i.e. penalization) parameter.
###
def _proximal_param_reg_matrix(X, lsig, gamma):
    # Parameter proximal mapping of the log-prior.
    sig = jnp.exp(-lsig)
    singular_values = jnp.linalg.svd(X, compute_uv=False)
    tr_norm = jnp.sum(jnp.maximum(singular_values-sig*gamma, 0))
    return tr_norm

@jax.jit
def _proximal_param_reg(X, lsig, gamma):
    # Parameter proximal mapping of the log-prior.
    tr_norm_vec = jax.vmap(_proximal_param_reg_matrix, in_axes=(2, None, None))(X, lsig, gamma)
    sig = jnp.exp(-lsig)
    tr_norm = jnp.sum(tr_norm_vec)
    return lsig + tfp.math.lambertw(gamma * sig * tr_norm)

@jax.jit
def ave_proximal_param_reg(X, lsig, gamma):
    """Parameter gradient averaged over particle cloud."""
    proximal = jax.vmap(_proximal_param_reg, in_axes=(3, None, None))(X, lsig, gamma)
    return proximal.mean()

###
# Gradient for the basis matrix Theta, common to all the matrices of the dataset.
###
def _grad_param_basis_matrix(X, mask, Y, Theta):
    aux = jnp.where(mask,jnp.dot(Theta, X), 0) - Y
    return jnp.dot(aux, X.T)

@jax.jit
def _grad_param_basis(X, mask, Y, Theta):
    # Parameter proximal mapping of the log-prior.
    grad_matrices = jax.vmap(_grad_param_basis_matrix, in_axes=(2, 2, 2, None), out_axes=2)(X, mask, Y, Theta)
    return jnp.sum(grad_matrices, axis = 2)

@jax.jit
def ave_grad_param_basis(X, mask, Y, Theta):
    """Parameter gradient averaged over particle cloud."""
    grad = jax.vmap(_grad_param_basis, in_axes=(3, None, None, None), out_axes=2)(X, mask, Y, Theta)
    return jnp.mean(grad, axis = 2)

###
# Proximal mapping for the particle cloud (featuring the latent sparse matrices of the dataset).
###
def _proximal_particle(B, lsig, gamma):
    # Parameter proximal mapping of one of the two log-priors (of Laplace form) for the x argument.
    sig = jnp.exp(-lsig)
    u, singular_values, v = jnp.linalg.svd(B)
    singular_values_threshold = jnp.maximum(singular_values - sig * gamma, 0)
    n_padding = B.shape[1] - B.shape[0]
    Sigma_threshold = jnp.pad(jnp.diag(singular_values_threshold), ((0, 0), (0, n_padding)), mode='constant')  # we are going to assume that in general data is high-dimensional and wide
    return jnp.dot(jnp.dot(u, Sigma_threshold), v) 

@jax.jit
def _proximal_particle_matrix(X, lsig, gamma):
    # Parameter proximal mapping of the log-prior.
    return jax.vmap(_proximal_particle, in_axes=(2, None, None), out_axes=2)(X, lsig, gamma)

@jax.jit
def approx_proximal_particle(B, lsig, gamma):
    """w-gradient vectorized over particle cloud."""
    proximal = jax.vmap(_proximal_particle_matrix, in_axes=(3, None, None), out_axes=3)(B, lsig, gamma)
    return proximal

###
# Gradient for the particle cloud (featuring the latent sparse matrices of the dataset).
###
def _wgrad(X, mask, Y, Theta):
    aux = jnp.where(mask, jnp.dot(Theta, X), 0) - Y
    return jnp.dot(Theta.T, aux)

@jax.jit
def _wgrad_matrix(X, mask, Y, Theta):
    # Parameter proximal mapping of one of the two log-priors (of Laplace form) for the x argument.
    return jax.vmap(_wgrad, in_axes=(2, 2, 2, None), out_axes=2)(X, mask, Y, Theta)

@jax.jit
def wgrad(X, mask, Y, Theta):
    """w-gradient vectorized over particle cloud."""
    gradv = jax.vmap(_wgrad_matrix, in_axes=(3, None, None, None), out_axes=3)
    return gradv(X, mask, Y, Theta)


# Functions for the performance metrics.

def _nn(w, v, image):
    # Network's output when evaluated at image with weights w, v.
    arg = jnp.dot(v, jnp.tanh(jnp.dot(w, image.reshape((28**2)))))
    return jax.nn.softmax(arg)


def _nn_vec(w, v, images):
    # _nn vectorized over images.
    return jax.vmap(_nn, in_axes=(None, None, 0))(w, v, images)


def _nn_vec_vec(w, v, images):
    # _nn_vec vectorized over particles.
    return jax.vmap(_nn_vec, in_axes=(2, 2, None), out_axes=2)(w, v, images)


@jax.jit
def log_pointwise_predrictive_density(w, v, images, labels):
    """Returns LPPD for set of (test) images and labels."""
    s = _nn_vec_vec(w, v, images).mean(2)
    return jnp.log(s[jnp.arange(labels.size), labels]).mean()


def _predict(w, v, images):
    # Returns label maximizing the approximate posterior predictive 
    # distribution defined by the cloud (w,v), vectorized over images.
    s = _nn_vec_vec(w, v, images).mean(2)
    return jnp.argmax(s, axis=1)


@jax.jit
def test_error(w, v, images, labels):
    """Returns fraction of misclassified images in test set."""
    return jnp.abs(labels - _predict(w, v, images)).mean()


def _mse(true_matrix, B):
    return jnp.sum((B - true_matrix) ** 2)/jnp.sum((true_matrix) ** 2)


@jax.jit
def mean_square_error(true_matrix, B):
    """Returns fraction of misclassified images in test set."""
    mse = jax.vmap(_mse, in_axes=(None, 3))(true_matrix, B)
    return mse.mean()


def _mse_missing(true_matrix, B, mask):
    return jnp.sum((jnp.where(mask, 0, B) - jnp.where(mask, 0, true_matrix)) ** 2)/jnp.sum((jnp.where(mask, 0, true_matrix)) ** 2)


@jax.jit
def mean_square_error_missing(true_matrix, B, mask):
    """Returns fraction of misclassified images in test set."""
    mse = jax.vmap(_mse_missing, in_axes=(None, 3, None))(true_matrix, B, mask)
    return mse.mean()