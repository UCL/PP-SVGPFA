function [dKhprs] = dKhprs_PeriodicKernel(prs,X1,varargin)

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

G = PeriodicKernel(prs,X1,X2);

dd = bsxfun(@minus,X1,permute(X2,[2 1 3]));
rr = (pi.*dd./period);

dKhprs(:,:,1,:) = permute(2*G/variance,[1 2 4 3]);
dKhprs(:,:,2,:) = permute(4*G/lengthscale^3 .*sin(rr).^2,[1 2 4 3]);
dKhprs(:,:,3,:) = permute(4*G*pi.*dd/period^2.*sin(rr)/lengthscale^2 .*cos(rr),[1 2 4 3]);
