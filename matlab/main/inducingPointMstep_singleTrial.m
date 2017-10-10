function prs = inducingPointMstep_singleTrial(m,ntr)

% extract hyperparameters and inducing points for each latent process
zz = cell2mat(m.Z);
prs0 = zz(:,:,ntr);

% make objective function
fun = @(prs) inducingPointMstep_singleTrial_Objective(m,prs,ntr);
% DerivCheck(fun,prs0);
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

prs = minFunc(fun,prs0,optimopts);

end