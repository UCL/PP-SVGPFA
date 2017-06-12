function L = stable_chol(K);

% abstol = eps*max(size(K));
tol = 1e-06;

% [U,S,~] = svd(K);
% ss = diag(S);
% if any(ss < tol)
%     ss(ss < tol) = min(ss(ss > max(ss(ss < tol))));
%     K = U*diag(ss)*U';
% end
L = chol(K+ tol*eye(size(K)));