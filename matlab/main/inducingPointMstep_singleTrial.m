function prs = inducingPointMstep_singleTrial(m,ntr)

% extract hyperparameters and inducing points for each latent process
prs0 = cell2mat(m.Z(:,ntr));

% make objective function
fun = @(prs) inducingPointMstep_Objective(m,prs,ntr);

% run optimizer
opts = optimset('Gradobj','on','display', 'none');
opts.MaxIter = 20;
prs = minFunc(fun,prs0,opts);

end