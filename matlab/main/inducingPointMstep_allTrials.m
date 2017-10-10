function prs = inducingPointMstep_allTrials(m);

prs0 = vec(cell2mat(m.Z));

% make objective function
fun = @(prs) inducingPointMstep_Objective(m,prs);
% DerivCheck(fun,prs0);
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

prs = minFunc(fun,prs0,optimopts);
