function Ki = pinv_ridge(K);
% function to compute pseudoinverse with added ridge.

tol = 1e-06;
Ki= pinv(K + tol*eye(size(K)));

% [U,S] = svd(K);
% thresh = 1e12;  % threshold on condition number
% sdiag = diag(S); 
% ii = max(sdiag)./sdiag < thresh;  % indices to keep
% 
% ss = 1./sdiag;
% ss(~ii) = 0;
% Ki = U*diag(ss)*U';