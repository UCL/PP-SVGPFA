function grad = gradLik_hprs_Poisson_svGPFA(m,Kmats,idxhprs,varargin);

nhprs = sum(cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0)));
grad_lik_poi1 = zeros(nhprs,m.ntr);
grad_lik_poi2 = zeros(nhprs,m.ntr);

mu_h = varargin{1};
var_h = varargin{2};
mask = permute(repmat(m.mask,[1 1 m.dy]),[1 3 2]); % check this
intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
intval(mask) = 0;

R = max(m.trLen);

[ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_posteriorGP_svGPFA(m,Kmats,R);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_hprs{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_hprs{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    
    grad_lik_poi1(idxhprs{k},:) = permute(m.BinWidth*sum(reshape(bsxfun(@times,permute(intval,[1 2 4 3]),logExp),[],m.kerns{k}.numhprs,m.ntr),1),[2 3 1]);
    
    grad_lik_poi2(idxhprs{k},:) = permute(mtimesx(mtimesx(m.prs.C(:,k)',m.Y),ddk_mu_hprs{k}),[2 3 1]);
end

grad = sum(-grad_lik_poi1 + grad_lik_poi2,2);
