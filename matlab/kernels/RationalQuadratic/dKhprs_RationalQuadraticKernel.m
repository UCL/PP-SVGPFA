function [dKhprs] = dKhprs_RationalQuadraticKernel(prs,X1,varargin)
%
% gradient with respect to hyperparameters
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
rr = (1 + ddist/(2*alpha^2*lengthscale^2));
G = variance^2*rr.^(-alpha^2);

dKhprs(:,:,1,:) = 2*G/variance;
dKhprs(:,:,2,:) = - variance^2*rr.^(-alpha^2 -1).*ddist/(alpha^2*lengthscale^3);
dKhprs(:,:,3,:) = G.*((ddist/lengthscale^2)./(alpha*rr) - 2*alpha*log(rr));

