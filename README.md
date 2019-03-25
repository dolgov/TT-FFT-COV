# TT-FFT-COV
Field generation and kriging using circulant covariance matrix and Tensor Train (TT) approximations
for the paper "Kriging in Tensor Train data format" by Sergey Dolgov, Alexander Litvinenko and Dishi Liu.

This package consists of two drivers:
 * driver_generate_field.m: generates a random field given a circulant covariance matrix, using superposition of a Cartesian product of iid normal vectors.
 * driver_cov_fft.m: interpolation of a set of samples onto a coarse Cartesian grid and kriging onto a fine grid via circulant covariance matrices.

TT algorithms use TT-Toolbox (https://github.com/oseledets/TT-Toolbox).
It can be downloaded automatically using the check_tt_install script.

