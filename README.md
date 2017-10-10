# PP-SVGPFA

Point Process - Sparse Variational Gaussian Process Factor Analysis

This repository contains code to implement various algorithms
to perform inference and learning in GPFA models.

## Models treated:
* GPFA with Poisson and Gaussian observations
* GPFA with Point Process observations


## Algorithms:
* VGPFA: Variational GPFA
* SVGPFA: Sparse Variational GPFA
* pp-SVGPFA: Point Process Sparse Variational GPFA

The most relevant available reference for the sparse additive algorithms is:
```
@inproceedings{adam2016scalable,
  title={Scalable transformed additive signal decomposition by non-conjugate Gaussian process inference},
  author={Adam, Vincent and Hensman, James and Sahani, Maneesh},
  booktitle={Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
```

## Python

### Requirements (Python 3)
* tensorflow==1.1.0
* numpy==1.12.1
* matplotlib==2.0.2


The python implementation heavily relies on [GPflow](https://github.com/GPflow/GPflow)


### Demonstration

Demonstration scripts for both (S)VGPFA and PP-SVGPFA can be found in `python/scripts/`
These consist in inference and learning on synthetic examples.


## Matlab
### Requirements (MATLAB R2016b)
* [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) (included in /matlab/misc/)
* [mtimesx](https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support) (included in /matlab/misc/)

These contain mex files which need to be compiled in order for the code to work

### Demonstration

A demonstration scripts for a synthetic data example can be found in demo.m
