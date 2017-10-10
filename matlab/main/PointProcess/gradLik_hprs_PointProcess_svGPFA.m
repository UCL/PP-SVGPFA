function grad = gradLik_hprs_PointProcess_svGPFA(m,Kmats,idxhprs,varargin);

nhprs = sum(cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0)));
grad_lik_pp1 = zeros(nhprs,m.ntr);
grad_lik_pp2 = zeros(nhprs,m.ntr);

mu_h = varargin{1};
mu_hQuad = mu_h{1};
var_h = varargin{2};

intval = exp(mu_hQuad + 0.5*var_h); % T x N x length(trEval)

R = max(m.trLen);

[ddk_mu_hprs,ddk_sig_hprs,ddk_mu_hprsObs] = grads_hprs_posteriorGP_svGPFA(m,Kmats,R);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_hprs{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_hprs{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    grad_lik_pp1(idxhprs{k},:) = permute(sum(mtimesx(permute(m.wwQuad,[2 1 4 3]),bsxfun(@times,permute(intval,[1 2 4 3]),logExp)),2),[3 4 2 1]);
    grad_lik_pp2(idxhprs{k},:) = cell2mat(ddk_mu_hprsObs{k}')';
    
end
    
grad = sum(-grad_lik_pp1 + grad_lik_pp2,2);


