function [prs,objval] = Estep_singleTrial(m,ntr);
% function to run E-steps for individual trials

% pre-allocate space for kernel matrices
allKzzi = cell(m.dx,1);
allKtz = cell(m.dx,1);
allKtt = cell(m.dx,1);
allKtzObs = cell(m.dx,1);

% pre-compute kernel matrices
for k = 1:m.dx
    Z_k = m.Z{k,ntr};
    % small ridge added to Kzz diagonal to avoid numerical issues with inverses:
    Kzz = m.kerns{k}.K(m.kerns{k}.hprs,Z_k) + m.epsilon*eye(m.num_inducing(k));
    allKzzi{k} = pinv(Kzz);
    
    allKtz{k} = m.kerns{k}.K(m.kerns{k}.hprs,m.tt,Z_k);
    allKtt{k} = m.kerns{k}.Kdiag(m.kerns{k}.hprs,m.tt);
    
    allSpikes = cellvec(m.Y(:,ntr));
    neuronIndex = m.neuronIndex{ntr};
    allKtzObs{k} = m.kerns{k}.K(m.kerns{k}.hprs,allSpikes,Z_k);
    
end

% extract E step parameters to be optimized
mu0 = [];
sig0= [];
sigdiag0 = [];

for k = 1:m.dx
    mu0 = [mu0; m.q_mu{k}(:,:,ntr)];
    sig0 = [sig0;m.q_sqrt{k}(:,:,ntr)];
    sigdiag0 = [sigdiag0;m.q_diag{k}(:,:,ntr)];
end

% make objective function handle
prs0 = [mu0;sig0;sigdiag0];
fun = @(prs) Estep_Objective(m,prs,allKzzi,allKtz,allKtt,allKtzObs,neuronIndex);

% run optimizer
opts = optimset('Gradobj','on','display', 'none');
opts.MaxIter = 100;
prs = minFunc(fun,prs0,opts);

objval = fun(prs);

end
     