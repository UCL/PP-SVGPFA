This repository contains MATLAB code to run sparse variational Gaussian Process Factor Analysis (svGPFA).

##supported likelihoods:
* Gaussian
* Poisson (exponential non-linearity)
* Point Process (exponential non-linearity)

##supported covariance functions
* RBF
* Periodic
* Locally Periodic
* Matern 3/2
* Matern 5/2
* Rational Quadratic

The code contains two useful packages for the optimisation routines and for faster computation of tensor products in MATLAB.
Thanks to: Mark Schmidt (minFunc https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) and James Tursa (mtimesx https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support) for making these available. 

We include these packages in here to make our code self-contained. 

Note that these contain mex files which need to be compiled and set up in order for the code to work properly. 
See documentation in the respective directories for more detail.
