function [mu_k, var_k] = predict_single(m,Tnew,k,ntr,varargin)
% predictive distributition of single GP function at time points
%
Z_k = m.Z{k,ntr};
q_mu_k = m.q_mu{k}(:,:,ntr);
% q_sqrt_k = MakeTriangular(m.q_sqrt{k,ntr});
q_sqrt_k = reshape(m.q_sqrt{k}(:,:,ntr),m.num_inducing(k),m.rnk(k));
q_diag_k = m.q_diag{k}(:,:,ntr);
q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);

if nargin < 5
    Ktz = m.kerns{k}.K(m.kerns{k}.hprs,Tnew,Z_k);
    Ktt = m.kerns{k}.Kdiag(m.kerns{k}.hprs,Tnew);
else
    Ktz = varargin{2};
    Ktt = varargin{3};
end
Kzz = m.kerns{k}.K(m.kerns{k}.hprs,Z_k) + m.epsilon*eye(m.num_inducing(k));
Lzz = chol(Kzz);
Ak = solve_chol(Lzz,Ktz');
mu_k= Ak'*q_mu_k;
var_k= (Ktt +  sum(Ak.*((q_sigma_k - Kzz)*Ak))');

end    

       


