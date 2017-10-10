function grad = gradLik_inducingPoints_Poisson_svGPFA(m,Kmats,idxZ,trEval,varargin);

grad_lik_poi1 = zeros(sum(m.numZ),length(trEval));
grad_lik_poi2 = zeros(sum(m.numZ),length(trEval));

mu_h = varargin{1};
var_h = varargin{2};

mask = permute(repmat(m.mask(:,trEval),[1 1 m.dy]),[1 3 2]); % check this
intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
intval(mask) = 0;

R = max(m.trLen);

[ddk_mu_in,ddk_sig_in] = grads_inducingPoints_posteriorGP_svGPFA(m,Kmats,trEval,R);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_in{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_in{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    
    grad_lik_poi1(idxZ{k},:) = permute(m.BinWidth*sum(reshape(bsxfun(@times,permute(intval,[1 2 4 3]),logExp),[],m.numZ(k),length(trEval)),1),[2 3 1]);   
    
    grad_lik_poi2(idxZ{k},:) = permute(mtimesx(mtimesx(m.prs.C(:,k)',m.Y(:,:,trEval)),ddk_mu_in{k}),[2 3 1]);
end

grad = - grad_lik_poi1(:) + grad_lik_poi2(:);

