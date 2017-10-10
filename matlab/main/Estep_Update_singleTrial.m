function prs = Estep_Update_singleTrial(m,Kmats,ntr);
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

current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
% extract Z's for current trial
current_Z = cellfun(@(x) x(:,:,ntr),m.Z,'uni',0);
Kmats = m.EMfunctions.BuildKernelMatrices(m,current_hprs,current_Z,0,ntr);
% make objective function
fun = @(prs) Estep_singleTrial_Objective(m,prs,Kmats,ntr);
% DerivCheck(fun,prs0);
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.Estep;

prs = minFunc(fun,prs0,optimopts);
