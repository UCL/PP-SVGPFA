function grad = gradLik_inducingPoints_PointProcess_svGPFA(m,Kmats,idxZ,trEval,varargin);

grad_lik_pp1 = zeros(sum(m.numZ),length(trEval));
grad_lik_pp2 = zeros(sum(m.numZ),length(trEval));

mu_h = varargin{1};
mu_hQuad = mu_h{1};
var_h = varargin{2};

intval = exp(mu_hQuad + 0.5*var_h); % T x N x length(trEval)

R = max(m.trLen);

[ddk_mu_in,ddk_sig_in,ddk_mu_inObs] = grads_inducingPoints_posteriorGP_svGPFA(m,Kmats,trEval,R);

for k = 1:m.dx
    
    logExp = bsxfun(@times,permute(ddk_mu_in{k},[1 5 2 3 4]),m.prs.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_in{k},[1 5 2 3 4]),m.prs.C(:,k).^2');
    
    grad_lik_pp1(idxZ{k},:) = permute(sum(mtimesx(permute(m.wwQuad(:,:,trEval),[2 1 4 3]),bsxfun(@times,permute(intval,[1 2 4 3]),logExp)),2),[3 4 2 1]);   
                              
    grad_lik_pp2(idxZ{k},:) = cell2mat(ddk_mu_inObs{k}')';
    
end

grad = - grad_lik_pp1(:) + grad_lik_pp2(:);
    
