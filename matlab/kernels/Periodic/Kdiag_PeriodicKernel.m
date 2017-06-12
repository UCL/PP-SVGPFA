function [Gdiag,dGdiag] = Kdiag_PeriodicKernel(prs,X1)
% computes only diagonal variance elements of kernel with equal
% inputs
variance = prs(1);
lengthscale = prs(2);
period = prs(3);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

Gdiag = variance^2*ones(size(X1));
dGdiag(:,:,1) = 2*Gdiag/variance;
dGdiag(:,:,2) = zeros(size(Gdiag));
dGdiag(:,:,3) = zeros(size(Gdiag));

