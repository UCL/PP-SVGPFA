function [obj, grad] = inducingPointMstep_Objective(m,prs)

% extract induincing points from parameter vector
Z = cell(m.dx,1);
idxZ = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

prs = reshape(prs,[],1, m.ntr);
for kk = 1:m.dx 
    Z{kk} = prs(istrt(kk):iend(kk),:,:);
%     idxZ{kk} = vec(bsxfun(@plus,(istrt(kk):iend(kk))',sum(m.numZ).*(0:(m.ntr-1))));
    idxZ{kk} = (istrt(kk):iend(kk));
end

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = m.EMfunctions.BuildKernelMatrices(m,current_hprs,Z,2);

[mu_h,var_h] = predict_MultiOutputGP_svGPFA(m,Kmats);

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h);
gradElik = m.EMfunctions.gradLik_inducingPoints(m,Kmats,idxZ,1:m.ntr,mu_h,var_h);

% get KL divergence and gradient
KLd = KL_div_svGPFA(m,Kmats);
gradKLd = KLgrad_inducingPoints_svGPFA(m,Kmats,idxZ,1:m.ntr);

obj = - Elik + KLd; % negative free energy
grad = - gradElik + gradKLd; % gradients