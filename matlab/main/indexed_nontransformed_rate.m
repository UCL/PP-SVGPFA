function mu_h_n = indexed_nontransformed_rate(m,nidx,mu_k)
% map conditionals to firing rate
Cn = m.C(nidx,:);
bn = m.b(nidx);
mu_h_n = sum(mu_k.*Cn,2) + bn; % length(Tnew)x1 vector
end
