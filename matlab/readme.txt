This repository contains MATLAB code to run sparse variational point process Gaussian Process Factor Analysis (sv-ppGPFA).

The code contains two useful packages for the optimisation routines and for faster computation of tensor products in MATLAB.

Thanks to:

- Mark Schmidt (minFunc https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
- James Tursa (mtimesx https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support)

for making these available. 

We include these packages in here to make our code self-contained. 

Note that these contain mex files which need to be compiled and set up in order for the code to work properly. 
See documentation in the respespective directories for more detail.
