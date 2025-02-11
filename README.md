# PIP(install)LA: PROXIMAL INTERACTING PARTICLE LANGEVIN ALGORITHMS

[![Python 3.8 - 3.11](https://img.shields.io/badge/Python-3.8%20--%203.11-blue)](https://www.python.org/downloads/release/python-3113/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/license/mit/)

Repository with Proximal Interacting Particle Langevin Algorithms, to perform learning and inference in latent variable models with non-differentiable joint density.


**Description**

This repository contains code illustrating the application of the different proximal interacting particle Langevin algorithms. 
We show the performance of our methods in a toy hierarchical model, a Bayesian logistic regression problem with sparsity inducing prior, a BNN classifier in MNIST data with a Laplace prior on the weights or with non-differentiable activation functions, an image deblurring experiment and a finally a matrix completion problem. 


## Requirements 

PIPLA requires the following Python packages:
* `cv2` 
* `imageio` 
* `jax`
* `matplotlib`
* `numpy`
* `pickle`
* `prox_tv`
* `scipy`
* `skimage.metrics`
* `tensorflow_probability`
* `torch`
* `tqdm`
    

