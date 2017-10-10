function [mu_h,var_h,mu_k,var_k] = predict_MultiOutputGP_svGPFA(m,Kmats,ntr);

if nargin < 3
    [mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats);
    trEval = 1:m.ntr;
else
    [mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats,ntr);
    trEval = ntr;
end

% include helper indices to keep track of changing structure depending on
% parallel or sequential updating
if length(trEval) > 1 % evalutate over all trials
    timeEval = trEval;
elseif length(trEval) == 1
    timeEval = 1;
end


if iscell(mu_k) % applies to point process version
    mu_h{1} = bsxfun(@plus,mtimesx(mu_k{1},m.prs.C'), m.prs.b');
    for nn = 1:length(trEval)
        mu_h{2,nn} = bsxfun(@plus,sum(bsxfun(@times,mu_k{2,timeEval(nn)},m.prs.C(m.index{trEval(nn)},:)),2), m.prs.b(m.index{trEval(nn)}));
    end
else
    mu_h = bsxfun(@plus,mtimesx(mu_k,m.prs.C'), m.prs.b');
end
var_h = mtimesx(var_k,(m.prs.C.^2)');