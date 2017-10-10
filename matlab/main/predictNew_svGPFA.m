function pred = predictNew_svGPFA(m,testTimes);

Kmats = KernelMatrices_prediction_svGPFA(m,testTimes);
[mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats);

mu_h = bsxfun(@plus,mtimesx(mu_k,m.prs.C'), m.prs.b');
var_h = mtimesx(var_k,(m.prs.C.^2)');

pred.latents.mean = mu_k;
pred.latents.variance = var_k;
pred.multiOutputGP.mean = mu_h;
pred.multiOutputGP.variance = var_h;