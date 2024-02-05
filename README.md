# Mixed Noise and Posterior Estimation with Conditional DeepGEM

## Problem Description and related work

Code accompanying the paper "Mixed Noise and Posterior Estimation with Conditional DeepGEM". Here we learn the noise parameters $(a,b)$ of a Bayesian inverse problem $Y = f(X) + \xi,$ 
where $\xi \sim  \mathcal{N}\big(0, a^2 + b^2 \text{diag} (f(x)^2) \big)$ is a mixture of additive and multiplicative Gaussian noise. This approach allows to incorporate information from several measurements within one model 
and therefore for more accurate reconstruction of the noise parameters, when more than one measurement is available, see the image below. Here the distance to the true a and b decreases as more information is given. 
<p align="center">
<img src =https://github.com/PaulLyonel/ConditionalDeepGEM/assets/16451311/0837d9f5-45f1-4423-b2fb-07ddd2b86452>
</p>
We build upon the deepGEM framework [2], and combine it with conditional normalizing flows [3]
to solve scatterometric inverse problems [4,5]. 
If you use the forward models from the low-dimensional photo mask please cite the corresponding papers [4,5]. if you use the forward model from the oxide layer please cite the 
zenodo page [6] and our paper [1].

## Code 

This code contains 4 jupyter notebooks. The ones on the main branch are for the more high dimensional oxide layer example and use the forward KL (EM_scatter) and the reverse KL (EM_deepgem). Both of these use synthetic measurements
created via the surrogate forward models. The lower dimensional photo mask examples are to be found in this subfolder. 


## Links

[1] Mixed Noise and Posterior Estimation with Conditional DeepGEM, Hagemann et al

[2] DeepGEM: Generalized Expectation-Maximization for Blind Inversion, Gao et al, NeurIPS 2021

[3] Guided Image Generation with Conditional Invertible Neural Networks, Ardizzone et al, arXiv 1907.02392

[4] Bayesian approach to the statistical inverse problem of scatterometry: Comparison of three surrogate models, Heidenreich et al, International Journal for Uncertainty Quantification, 5(6), 2015

[5] Bayesian approach to determine critical dimensions from scatterometric measurements, Heidenreich et al, Metrologia, 55(6):S201, 2018

[6] Zenodo [link](https://zenodo.org/records/10580011?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhODczMTI[…]Wy_BlNwAypimG3ogbySLuIiCMvye4__sW6wKv4jSbj46saixcrQuZPugke0w5aw)
## Citation 
