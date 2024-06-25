import os
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch 
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##############################################################
## PROXIMAL PARTICLE ALGORITHMS
##############################################################

def pip_myula_bnn(ltrain, itrain, ltest, itest, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles.

    # Initialize arrays storing performance metrics as a function of k:
    lppd = np.zeros(K)  # Log pointwise predictive density (LPPD).
    error = np.zeros(K)  # Test error.

    for k in (tqdm(range(K))): 
        # Evaluate metrics for current particle cloud:
        lppd[k] = log_pointwise_predrictive_density(w, v, itest, ltest)
        error[k] = test_error(w, v, itest, ltest)

        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k]* (1-h/gamma) + h*ave_proximal_param(wk, a[k], gamma)/(Dw*gamma)- 2*h 
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Alpha.
        b = np.append(b, b[k]* (1-h/gamma) + h*ave_proximal_param(vk, b[k], gamma)/(Dv*gamma)- 2*h 
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Beta.

        # Update particle cloud:
        w = (w * (1-h/gamma) + h*wgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(wk, a[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v * (1-h/gamma) + h*vgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(vk, b[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v, lppd, error


def my_pgd_bnn(ltrain, itrain, ltest, itest, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles.

    # Initialize arrays storing performance metrics as a function of k:
    lppd = np.zeros(K)  # Log pointwise predictive density (LPPD).
    error = np.zeros(K)  # Test error.

    for k in (tqdm(range(K))): 
        # Evaluate metrics for current particle cloud:
        lppd[k] = log_pointwise_predrictive_density(w, v, itest, ltest)
        error[k] = test_error(w, v, itest, ltest)

        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k]* (1-h/gamma) + h*ave_proximal_param(wk, a[k], gamma)/(Dw*gamma) -2 * h)  # Alpha.
        b = np.append(b, b[k]* (1-h/gamma) + h*ave_proximal_param(vk, b[k], gamma)/(Dv*gamma) -2 * h)  # Beta.

        # Update particle cloud:
        w = (w * (1-h/gamma) + h*wgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(wk, a[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v * (1-h/gamma) + h*vgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(vk, b[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v, lppd, error


def pip_myula_bnn_performance(ltrain, itrain, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles.

    for k in (tqdm(range(K))): 
        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k]* (1-h/gamma) + h*ave_proximal_param(wk, a[k], gamma)/(Dw*gamma) -2 * h
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Alpha.
        b = np.append(b, b[k]* (1-h/gamma) + h*ave_proximal_param(vk, b[k], gamma)/(Dv*gamma) -2 * h
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Beta.

        # Update particle cloud:
        w = (w * (1-h/gamma) + h*wgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(wk, a[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v * (1-h/gamma) + h*vgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(vk, b[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v


def my_pgd_bnn_performance(ltrain, itrain, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles.

    for k in (tqdm(range(K))): 
        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k]* (1-h/gamma) + h*ave_proximal_param(wk, a[k], gamma)/(Dw*gamma) -2 * h)  # Alpha.
        b = np.append(b, b[k]* (1-h/gamma) + h*ave_proximal_param(vk, b[k], gamma)/(Dv*gamma) -2 * h)  # Beta.

        # Update particle cloud:
        w = (w * (1-h/gamma) + h*wgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(wk, a[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v * (1-h/gamma) + h*vgrad(wk, vk, itrain, ltrain) + h * approx_proximal_particle(vk, b[k], gamma)/gamma
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v


##############################################
# AUXILIARY FUNCTIONS
##############################################

# FUNCTIONS FOR THE LOG DENSITY

def _log_nn(w, v, image):
    # Log of the network's output when evaluated at image with weights w, v.
    arg = jnp.dot(v, jnp.tanh(jnp.dot(w, image.reshape((28**2)))))
    return jax.nn.log_softmax(arg)

def _log_nn_vec(w, v, images):
    # _log_nn vectorized over particles.
    return jax.vmap(_log_nn, in_axes=(None, None, 0))(w, v, images)


def _log_likelihood(w, v, images, labels):
    # Log-likelihood for set of images and labels, vectorized over particles.
    return (_log_nn_vec(w, v, images)[jnp.arange(labels.size), labels]).sum()



# FUNCTIONS FOR THE GRADIENTS OF THE LOG-DENSITY

def _grad_param(x, lsig):
    # Parameter gradient of one of the two log-priors.
    v = x.reshape((x.size))
    sig = jnp.exp(lsig)
    return jnp.dot(v, v)/(sig**2) - x.size

@jax.jit
def ave_grad_param(w, lsig):
    """Parameter gradient averaged over particle cloud."""
    grad = jax.vmap(_grad_param, in_axes=(2, None))(w, lsig)
    return grad.mean()

def _proximal_param(x, lsig, gamma):
    # Parameter proximal mapping of one of the two log-priors (of Laplace form).
    b = 2 * gamma * jnp.sum(jnp.abs(x))
    sig = jnp.exp(lsig)
    return lsig + 0.5 * tfp.math.lambertw(2*b/(sig**2))

@jax.jit
def ave_proximal_param(w, lsig, gamma):
    """Parameter gradient averaged over particle cloud."""
    proximal = jax.vmap(_proximal_param, in_axes=(2, None, None))(w, lsig, gamma)
    return proximal.mean()

def _proximal_particle(x, lsig, gamma):
    # Parameter proximal mapping of one of the two log-priors (of Laplace form) for the x argument.
    sig = jnp.exp(lsig)
    aux = (x-jnp.sign(x)*gamma/(sig**2))
    b = gamma/(sig**2)
    return aux * (jnp.abs(x)>=b)

@jax.jit
def approx_proximal_particle(w, lsig, gamma):
    """w-gradient vectorized over particle cloud."""
    proximal = jax.vmap(_proximal_particle, in_axes=(2, None, None), out_axes=2)(w, lsig, gamma)
    return proximal

@jax.jit
def wgrad(w, v, images, labels):
    """w-gradient vectorized over particle cloud."""
    grad = jax.grad(_log_likelihood, argnums=0)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None), out_axes=2)
    return gradv(w, v, images, labels)


@jax.jit
def vgrad(w, v, images, labels):
    """v-gradients vectorized over particle cloud."""
    grad = jax.grad(_log_likelihood, argnums=1)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None), out_axes=2)
    return gradv(w, v, images, labels)


# FUNCTIONS FOR THE PERFORMANCE METRICS

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
    """Returns label maximizing the approximate posterior predictive 
    distribution defined by the cloud (w,v), vectorized over images."""
    s = _nn_vec_vec(w, v, images).mean(2)
    return jnp.argmax(s, axis=1)


@jax.jit
def test_error(w, v, images, labels):
    """Returns fraction of misclassified images in test set."""
    return jnp.abs(labels - _predict(w, v, images)).mean()
