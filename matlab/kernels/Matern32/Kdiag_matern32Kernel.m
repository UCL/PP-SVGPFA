function [Gdiag,dGdiag] = Kdiag_matern32Kernel(prs,X1)
% radial basis function kernel, aka exponentiated quadratic aka squared
% exponential
%
% hyperparameters
variance = prs(1);
lengthscale = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end
% computes only diagonal variance elements of kernel with equal
% inputs
Gdiag = variance^2*ones(size(X1));
dGdiag(:,:,1,:) = permute(2*Gdiag/variance,[1 2 4 3]);
dGdiag(:,:,2,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
end