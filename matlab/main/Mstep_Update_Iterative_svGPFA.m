function m = Mstep_Update_Iterative_svGPFA(m,Kmats);

prs0 = [m.prs.C(:);m.prs.b];

% predict posterior means and variances
[mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats);

% make objective function
fun = @(prs) Mstep_Objective(m,prs,mu_k,var_k);
% DerivCheck(fun,prs0);
optimopts = optimset('Gradobj','on','display', 'none');

optimopts.MaxIter = m.opts.maxiter.Mstep;
prs = minFunc(fun,prs0,optimopts);

m.prs.C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
m.prs.b = prs(m.dy*m.dx + 1 : end);