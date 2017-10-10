function fe = VariationalFreeEnergy_svGPFA(m,Kmats,varargin);
% compute the variational free energy
[mu_h,var_h] = predict_MultiOutputGP_svGPFA(m,Kmats);

Elik = m.EMfunctions.likelihood(m,mu_h,var_h);

% get KL divergence 
KLd = KL_div_svGPFA(m,Kmats);

fe = Elik - KLd;