function G = RationalQuadraticKernel(prs,X1,varargin)
% rational quadratic covariance function
%
% hyperparameters
variance = prs(1);
lengthscale = prs(2);
alpha = prs(3);

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
G = variance^2*(1 + ddist/(2*alpha^2*lengthscale^2)).^(-alpha^2);

end

