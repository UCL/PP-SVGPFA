function [Gdiag,dGdiag] = Kdiag_LocallyPeriodicKernel(prs,X1)
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
% computes only diagonal variance elements of kernel with equal
% inputs
Gdiag = variance^2*ones(size(X1));
dGdiag(:,:,1,:) = permute(2*Gdiag/variance,[1 2 4 3]);
dGdiag(:,:,2,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
dGdiag(:,:,3,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
dGdiag(:,:,4,:) = permute(zeros(size(Gdiag)),[1 2 4 3]);
end