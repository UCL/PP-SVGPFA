function [dKhprs] = dKhprs_LocallyPeriodicKernel(prs,X1,varargin)
%
% gradient with respect to hyperparameters
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

dd = bsxfun(@minus,X1,permute(X2,[2 1 3]));
rr = (pi.*dd./period_per);

dKhprs(:,:,1,:) = 2*G1.*G2/variance;
dKhprs(:,:,2,:) = G1.*G2/lengthscale_se^3 .* dd.^2;

dKhprs(:,:,3,:) = permute(4*G1.*G2/lengthscale_per^3 .*sin(rr).^2,[1 2 4 3]);
dKhprs(:,:,4,:) = permute(4*G1.*G2*pi.*dd/period_per^2.*sin(rr)/lengthscale_per^2 .*cos(rr),[1 2 4 3]);
