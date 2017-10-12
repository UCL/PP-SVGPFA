function [dGin2, dGin1] = dKin_matern52Kernel(prs,X1,varargin)

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

rr  = abs(bsxfun(@minus,X1,permute(X2,[2 1 3])));

[N1,N2,ntr] = size(rr);

dGin2 = zeros(N1,N2,N2,ntr);

if nargout > 1
    dGin1 = zeros(N1,N2,N1,ntr);
end

ColMask = reshape(full(logical(kron(speye(N2),ones(N1,1)))),[N1 N2 N2]);
ColMask = repmat(ColMask,[1 1 1 ntr]);
dGin2(ColMask) = variance^2*(5/3*rr/lengthscale^2 + 5*sqrt(5)/3*rr.^2/lengthscale^3).*exp(-sqrt(5)*rr/lengthscale);

% gradient with respect to input points X2
if nargout > 1
    RowMask = reshape(full(logical(kron(speye(N1),ones(1,N2)))),[N1 N2 N1]);
    RowMask = repmat(RowMask,1,1,1,ntr);
    if nargin == 2 % grad wrt to first input if same
        dGin1(RowMask) = permute( -variance^2*(5/3*rr/lengthscale^2 + 5*sqrt(5)/3*rr.^2/lengthscale^3).*exp(-sqrt(5)*rr/lengthscale),[2 1 3 4]);
        dGin1(ColMask) = variance^2*(5/3*rr/lengthscale^2 + 5*sqrt(5)/3*rr.^2/lengthscale^3).*exp(-sqrt(5)*rr/lengthscale);
    else % grad wrt first input if different
        dGin1(RowMask) = -variance^2*(5/3*rr/lengthscale^2 + 5*sqrt(5)/3*rr.^2/lengthscale^3).*exp(-sqrt(5)*rr/lengthscale);
    end
end



