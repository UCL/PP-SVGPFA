function G = rbfKernel(prs,X1,varargin)
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
ddist  = bsxfun(@minus,X1,permute(X2,[2 1 3])).^2;
G = variance^2*exp(-0.5*ddist/lengthscale^2);

end

