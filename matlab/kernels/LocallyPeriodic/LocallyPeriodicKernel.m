function G = LocallyPeriodicKernel(prs,X1,varargin)
% locally periodic kernel
%
% hyperparameters
variance = prs(1);
lengthscale_se = prs(2);
lengthscale_per = prs(3);
period_per = prs(4);

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
G1 = rbfKernel([variance lengthscale_se],X1,X2);
G2 = PeriodicKernel([1 lengthscale_per period_per],X1,X2);
G = G1.*G2;

end

