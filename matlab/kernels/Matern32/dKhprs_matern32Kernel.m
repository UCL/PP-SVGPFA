function [dKhprs] = dKhprs_matern32Kernel(prs,X1,varargin)
%
% gradient with respect to hyperparameters
%
% hyperparameters
variance = prs(1);
lengthscale = prs(2);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% inputs
if nargin == 2
    X2 = X1;
else
    X2 = varargin{1};
    if isempty(X2)
        X2 = zeros(0,1);
    end
end
% returns kernel Gram matrix
rr  = abs(bsxfun(@minus,X1,permute(X2,[2 1 3])));
G = variance^2*(1 + sqrt(3)*rr/lengthscale).*exp(-sqrt(3)*rr/lengthscale);

dKhprs(:,:,1,:) = 2*G/variance;
dKhprs(:,:,2,:) =  variance^2*(3*rr.^2/lengthscale^3).*exp(-sqrt(3)*rr/lengthscale);

