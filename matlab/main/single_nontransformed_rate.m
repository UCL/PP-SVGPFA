function [mu_h_n,var_h_n] = single_nontransformed_rate(m,n,mu_k,varargin)
% map conditionals to firing rate
Cn = m.C.value(n,:);
bn = m.b.value(n);
mu_h_n = mu_k*Cn' + bn; % length(Tnew)x1 vector
if nargout > 1
    var_k = varargin{1};
    var_h_n = sum(bsxfun(@times,Cn.^2,var_k),2);
end
% we dont need off-diagonal terms
end
