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

############################################################################
## PROXIMAL PARTICLE ALGORITHMS FOR BNN WITH RELU-TYPE ACTIVATION FUNCTION
############################################################################

####################################
## MOREAU-YOSIDA IPLA
####################################
def my_ipla_bnn_activation(ltrain, itrain, ltest, itest, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles. 

    # Initialize arrays storing performance metrics as a function of k:
    lppd = np.zeros(K)  # Log pointwise predictive density (LPPD).
    error = np.zeros(K)  # Test error.

    for k in (tqdm(range(K))): 
        # Evaluate metrics for current particle cloud:
        lppd[k] = log_pointwise_predictive_density(w, v, itest, ltest)
        error[k] = test_error(w, v, itest, ltest)

        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k] + h*ave_grad_param(wk, a[k])/(Dw)
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Alpha.
        b = np.append(b, b[k] + h*ave_grad_param(vk, b[k])/(Dv)
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Beta.

        # Update particle cloud:
        aa, bb, cc = wprox(wk, vk, a[k], b[k], itrain, ltrain)
        print(bb, cc)
        print(aa.shape, np.sum(aa), "lets see")

        ee = wgrad(wk, vk, a[k], b[k], itrain, ltrain)
        print(ee.shape, "lets see")
        # print(aa.shape,aa)

        # print(b.shape, a.shape, wgrad(wk, vk, a[k], b[k], itrain, ltrain).shape, b)
        w = (w  + h*wgrad(wk, vk, a[k], b[k], itrain, ltrain) + h*aa #wprox(wk, vk, a[k], b[k], itrain, ltrain) 
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v + h*vgrad(wk, vk, a[k], b[k], itrain, ltrain) 
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v, lppd, error


def my_ipla_bnn_performance_activation(ltrain, itrain, h, K, a, b, w, v, gamma):
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

        a = np.append(a, a[k] + h*ave_grad_param(wk, a[k])/(Dw) 
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Alpha.
        b = np.append(b, b[k] + h*ave_grad_param(vk, b[k])/(Dv) 
                      + jnp.sqrt(2*h/N)*np.random.normal(0, 1, 1))  # Beta.

        # Update particle cloud:
        w = (w + h*wgrad(wk, vk, a[k], b[k], itrain, ltrain) + h*wprox(wk, vk, a[k], b[k], itrain, ltrain)
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        
        v = (v + h*vgrad(wk, vk, a[k], b[k], itrain, ltrain)
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v


##############################################
# PROXIMAL PARTICLE GRADIENT DESCENT ALGORITHM
##############################################

def my_pgd_bnn_activation(ltrain, itrain, ltest, itest, h, K, a, b, w, v, gamma):
    # Extract dimensions of latent variables:
    Dw = w[:, :, 0].size  # Dimension of w.
    Dv = v[:, :, 0].size  # Dimension of v.
    N = w.shape[-1] # Number of particles. 

    # Initialize arrays storing performance metrics as a function of k:
    lppd = np.zeros(K)  # Log pointwise predictive density (LPPD).
    error = np.zeros(K)  # Test error.

    for k in (tqdm(range(K))): 
        # Evaluate metrics for current particle cloud:
        lppd[k] = log_pointwise_predictive_density(w, v, itest, ltest)
        error[k] = test_error(w, v, itest, ltest)

        # Temporarily store current particle cloud:
        wk = w  # Input layer weights.
        vk = v  # Output layer weights.

        # Update parameter estimates (note that we are using the heuristic consisting on dividing the 
        # alpha-gradient by Dw and the beta-gradient by Dv):

        a = np.append(a, a[k] + h*ave_grad_param(wk, a[k])/(Dw))  # Alpha.
        b = np.append(b, b[k] + h*ave_grad_param(vk, b[k])/(Dv))  # Beta.

        # Update particle cloud:
        w = (w  + h*wprox(wk, vk, a[k], b[k], itrain, ltrain) 
               + jnp.sqrt(2*h) * np.random.normal(0, 1, w.shape)) 
        v = (v + h*vgrad(wk, vk, a[k], b[k], itrain, ltrain) 
               + jnp.sqrt(2*h) * np.random.normal(0, 1, v.shape))

    return a, b, w, v, lppd, error


##############################################
# AUXILIARY FUNCTIONS
##############################################

# FUNCTIONS FOR THE PROXIMAL OPERATOR

def h_nondiff(x):
    return jnp.where(x > 1, 1, jnp.where(x < -1, -1, x))

def prox_h_nondiff(x, lambdaa=0.001):
    """ReLU Proximity Map."""
    return jnp.where(x < -1 + lambdaa, 0, jnp.where(x <= 1 - lambdaa, 1, 0))

def _compute_expression(w, v, image, lambdaa=0.001):
    """Computes the given summation-based expression in JAX."""
    
    # Inner product of w and image
    hidden_product = jnp.dot(w, image.reshape((28**2))) # Shape (40,)

    # Compute the proximity function values
    prox_h_nondiff_values = prox_h_nondiff(hidden_product, lambdaa)  # Shape (40,)
    
    # COMPUTE THE FIRT TERM OF THE PROXIMAL TERM
    # (2, 40) @pointwise (40,) → (2, 40) then broadcasted with (1, 1, 784)
    term1 = (v * prox_h_nondiff_values[None, :])[:, :, None] * image.reshape(1, 1, 784)  # Shape (2, 40, 784)
    
    # To compute the second term
    # Compute exp_terms: (2, 40) @ (40,) → (2,)
    # exp_terms = jnp.exp(jnp.dot(v, h_nondiff(hidden_product)))  # Shape (2,)
    
    # Compute numerator: (2,) @ (2,40, 784) → (40,784)
    exp_terms_log = jnp.dot(v, h_nondiff(hidden_product))[None, :] - jnp.dot(v, h_nondiff(hidden_product))[:, None]
    #numerator = jax.scipy.special.logsumexp(exp_terms_log[:, None] + vlog, axis = 0)  
    #denominator = jax.scipy.special.logsumexp(exp_terms_log)
    #second_term = (jnp.exp(numerator - denominator) * prox_h_nondiff_values)[:, None] * image.reshape(1, 784)
    #second_term_expand = jnp.repeat(jnp.expand_dims(second_term, axis= 0), 2, axis=0)

    max_log = jnp.max(exp_terms_log)  # Find max value for stability
    stable_exp_terms_log = exp_terms_log - max_log  # Shift values down
    denominator = jnp.exp(-(jax.scipy.special.logsumexp(stable_exp_terms_log, axis = 0) + max_log))
    numerator = jnp.einsum("b, bmn -> mn", denominator, term1)
    second_term_expand = jnp.repeat(jnp.expand_dims(numerator, axis= 0), 2, axis=0) # Shape (2, 40, 784)
    
    #numerator = jnp.einsum("b, bmn -> mn", exp_terms, term1)  # Shape (40, 784)
    #numerator_expand = jnp.repeat(jnp.expand_dims(numerator, axis= 0), 2, axis=0) # Shape (2, 40, 784)

    # Compute denominator: scalar
    #denominator = jnp.sum(exp_terms)  # Shape ()

    prox_grad = term1 -second_term_expand #- numerator_expand / denominator # Shape (2, 40, 784)
    
    # create a tensor of the same dize as prox_grad_repeated
    # return prox_grad_repeated # Shape (2, 40, 784)
    return prox_grad, second_term_expand


def _prox_computation_vec(w, v, images, labels, lambdaa=0.001):
    # _log_nn vectorized over particles.
    _aux_prox, ee, oo = jax.vmap(_compute_expression, in_axes=(None, None, 0, None))(w, v, images, lambdaa)
    return (_aux_prox[jnp.arange(labels.size), labels, :, :]).sum(axis=0), ee, oo


@jax.jit
def wprox(w, v, a, b, images, labels, lambdaa=0.001):
    """w-gradient vectorized over particle cloud."""
    proxv = jax.vmap(_prox_computation_vec, in_axes=(2, 2, None, None, None), out_axes=2)
    return proxv(w, v, images, labels, lambdaa)


# FUNCTIONS FOR THE LOG DENSITY

def _log_nn(w, v, image):
    # Log of the network's output when evaluated at image with weights w, v.
    arg = jnp.dot(v, h_nondiff(jnp.dot(w, image.reshape((28**2)))))
    return jax.nn.log_softmax(arg)

def _log_nn_vec(w, v, images):
    # _log_nn vectorized over particles.
    return jax.vmap(_log_nn, in_axes=(None, None, 0))(w, v, images)

def _log_prior(x, lsig):
    # Log of a Gaussian prior, with mean 0 and variance e^lsig, evaluated at x.
    v = x.reshape((x.size))
    sig = jnp.exp(lsig)
    return -jnp.dot(v, v)/(2*sig**2) - x.size * (jnp.log(2*jnp.pi)/2 + lsig)


def _log_likelihood(w, v, images, labels):
    # Log-likelihood for set of images and labels, vectorized over particles.
    return (_log_nn_vec(w, v, images)[jnp.arange(labels.size), labels]).sum()


def _log_density(w, v, a, b, images, labels):
    # Log of model density, vectorized over particles.
    out = _log_prior(w, a) + _log_prior(v, b)
    return out + _log_likelihood(w, v, images, labels)


def _log_density_no_likelihood(w, v, a, b, images, labels):
    # Log of model density, vectorized over particles.
    out = _log_prior(w, a) + _log_prior(v, b)
    return out 


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


@jax.jit
def wgrad(w, v, a, b, images, labels):
    """w-gradient vectorized over particle cloud."""
    grad = jax.grad(_log_density_no_likelihood, argnums=0)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None, None, None), out_axes=2)
    return gradv(w, v, a, b, images, labels)


@jax.jit
def vgrad(w, v, a, b, images, labels):
    """v-gradients vectorized over particle cloud."""
    grad = jax.grad(_log_density, argnums=1)
    gradv = jax.vmap(grad, in_axes=(2, 2, None, None, None, None), out_axes=2)
    return gradv(w, v, a, b, images, labels)



# FUNCTIONS FOR THE PERFORMANCE METRICS

def _nn(w, v, image):
    # Network's output when evaluated at image with weights w, v.
    arg = jnp.dot(v, h_nondiff(jnp.dot(w, image.reshape((28**2)))))
    return jax.nn.softmax(arg)


def _nn_vec(w, v, images):
    # _nn vectorized over images.
    return jax.vmap(_nn, in_axes=(None, None, 0))(w, v, images)


def _nn_vec_vec(w, v, images):
    # _nn_vec vectorized over particles.
    return jax.vmap(_nn_vec, in_axes=(2, 2, None), out_axes=2)(w, v, images)


@jax.jit
def log_pointwise_predictive_density(w, v, images, labels):
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
