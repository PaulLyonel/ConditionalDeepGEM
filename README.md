# Mixed Noise and Posterior Estimation with Conditional DeepGEM

## Problem Description and related work

Code accompanying the paper "Mixed Noise and Posterior Estimation with Conditional DeepGEM". Here we learn the noise parameters $(a,b)$ of a Bayesian inverse problem $Y = f(X) + \xi,$ 
where $\xi \sim  \mathcal{N}\big(0, a^2 + b^2 \text{diag} (f(x)^2) \big)$ is a mixture of additive and multiplicative Gaussian noise. This approach allows to incorporate information from several measurements within one model 
and therefore for more accurate reconstruction of the noise parameters, when more than one measurement is available. Here the distance to the true a and b decreases as more information is given. 

We build upon the deepGEM framework [2], and combine it with conditional normalizing flows [3]
to solve scatterometric inverse problems [4,5]. 
If you use the forward models from the low-dimensional photo mask please cite the corresponding papers [4,5]. if you use the forward model from the oxide layer please cite the 
zenodo page [6] and our paper [1].

## Code 

The code is split up into reverse and forward KL for the high-dimensional version. For the lower dimensional scatterometry example, we also included baselines, where we chose a and b on a grid with comparable run time.

## Links

[1] Mixed Noise and Posterior Estimation with Conditional DeepGEM, Hagemann et al, arXiv:2402.02964

[2] DeepGEM: Generalized Expectation-Maximization for Blind Inversion, Gao et al, NeurIPS 2021

[3] Guided Image Generation with Conditional Invertible Neural Networks, Ardizzone et al, arXiv 1907.02392

[4] Bayesian approach to the statistical inverse problem of scatterometry: Comparison of three surrogate models, Heidenreich et al, International Journal for Uncertainty Quantification, 5(6), 2015

[5] Bayesian approach to determine critical dimensions from scatterometric measurements, Heidenreich et al, Metrologia, 55(6):S201, 2018

[6] Zenodo [link](https://zenodo.org/records/10580011?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhODczMTI[…]Wy_BlNwAypimG3ogbySLuIiCMvye4__sW6wKv4jSbj46saixcrQuZPugke0w5aw)
## Citation 
@article{Hagemann_2024,

doi = {10.1088/2632-2153/ad5926},

url = {https://dx.doi.org/10.1088/2632-2153/ad5926},

year = {2024},

month = {jul},

publisher = {IOP Publishing},

volume = {5},

number = {3},

pages = {035001},

author = {Paul Hagemann and Johannes Hertrich and Maren Casfor and Sebastian Heidenreich and Gabriele Steidl},

title = {Mixed noise and posterior estimation with conditional deepGEM},

journal = {Machine Learning: Science and Technology},

abstract = {We develop an algorithm for jointly estimating the posterior and the noise parameters in Bayesian inverse problems, which is motivated by indirect measurements and applications from nanometrology with a mixed noise model. We propose to solve the problem by an expectation maximization (EM) algorithm. Based on the current noise parameters, we learn in the E-step a conditional normalizing flow that approximates the posterior. In the M-step, we propose to find the noise parameter updates again by an EM algorithm, which has analytical formulas. We compare the training of the conditional normalizing flow with the forward and reverse Kullback–Leibler divergence, and show that our model is able to incorporate information from many measurements, unlike previous approaches.}
}

