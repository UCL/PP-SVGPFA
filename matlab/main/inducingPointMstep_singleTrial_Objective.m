function [obj, grad] = inducingPointMstep_singleTrial_Objective(m,prs,ntr)

% extract induincing points from parameter vector
Z = cell(m.dx,1);
idxZ = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

for kk = 1:m.dx
    Z{kk} = prs(istrt(kk):iend(kk));
    idxZ{kk} = (istrt(kk):iend(kk));
end

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = m.EMfunctions.BuildKernelMatrices(m,current_hprs,Z,2,ntr);

[mu_h,var_h] = predict_MultiOutputGP_svGPFA(m,Kmats,ntr);

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h,ntr);
gradElik = m.EMfunctions.gradLik_inducingPoints(m,Kmats,idxZ,ntr,mu_h,var_h);

% get KL divergence and gradient
KLd = KL_div_svGPFA(m,Kmats,ntr);
gradKLd = KLgrad_inducingPoints_svGPFA(m,Kmats,idxZ,ntr);

obj = - Elik + KLd; % negative free energy
grad = - gradElik + gradKLd; % gradients
