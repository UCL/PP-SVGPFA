function G = matern32Kernel(prs,X1,varargin)
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

end

