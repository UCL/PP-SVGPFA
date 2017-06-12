function G = PeriodicKernel(prs,X1,varargin)
variance = prs(1);
lengthscale = prs(2);
period = prs(3);

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

% returns kernel Gram matrix and derivative wrt hyperparameters
dd = bsxfun(@minus,X1,X2');

rr = (pi.*dd./period);
G = variance^2*exp(-2*sin(rr).^2/lengthscale^2);




