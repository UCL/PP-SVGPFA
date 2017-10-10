function prs = Estep_Update_allTrials(m,Kmats);

qmu0 = cell2mat(m.q_mu(:));
qsqrt0 = cell2mat(m.q_sqrt(:));
qdiag0 = cell2mat(m.q_diag(:));

prs0 = [qmu0;qsqrt0;qdiag0];
prs0 = prs0(:);
% make objective function
fun = @(prs) Estep_allTrials_Objective(m,prs,Kmats);
% DerivCheck(fun,prs0);
% run optimizer
optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = m.opts.maxiter.inducingPointMstep;

prs = minFunc(fun,prs0,optimopts);
