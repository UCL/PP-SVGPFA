function [mu_h, var_h] = map_to_nontransformed_rate(m,mu_k,var_k)
% calculate inferred firing rates

C = m.C;
b = m.b;
mu_h = bsxfun(@plus,mtimesx(mu_k,C'), b');
var_h = mtimesx(var_k,(C.^2)');

end