function [obj,grad] = hyperMstep_Objective(m,prs);
% to be passed into optimiser as function handle

% extract hprs from parameter vector
hprs = cell(m.dx,1);
idxhprs = cell(m.dx,1);

hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'uni',0)'));
istrthprs = [1; hprsidx(1:end-1)+1];
iendhprs = hprsidx;

for kk = 1:m.dx
    hprs{kk} = prs(istrthprs(kk):iendhprs(kk));
    idxhprs{kk} = (istrthprs(kk):iendhprs(kk));
end

% buil kernel matrices
Kmats = m.EMfunctions.BuildKernelMatrices(m,hprs,m.Z,1);

[mu_h,var_h] = predict_MultiOutputGP_svGPFA(m,Kmats);

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h);
gradElik = m.EMfunctions.gradLik_hprs(m,Kmats,idxhprs,mu_h,var_h);

% get KL divergence and gradient
KLd = KL_div_svGPFA(m,Kmats);
gradKLd = KLgrad_hprs_svGPFA(m,Kmats,idxhprs);

obj = -Elik + KLd; % negative free energy
grad = -gradElik + gradKLd; % gradients
