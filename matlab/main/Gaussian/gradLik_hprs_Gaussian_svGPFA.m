function grad = gradLik_hprs_Gaussian_svGPFA(m,Kmats,idxhprs,varargin);

nhprs = sum(cell2mat(cellfun(@(x)size(x,2),idxhprs,'uni',0)));
grad_lik_gg2 = zeros(nhprs,m.ntr);
grad_lik_gg3 = zeros(nhprs,m.ntr);

mu_h = varargin{1};
mask = permute(repmat(m.mask,[1 1 m.dy]),[1 3 2]); % check this
mu_h(mask) = 0;

R = max(m.trLen);

[ddk_mu_hprs,ddk_sig_hprs] = grads_hprs_posteriorGP_svGPFA(m,Kmats,R);

for k = 1:m.dx
    grad_lik_gg2(idxhprs{k},:) = -0.5*squeeze((1./m.prs.psi'*m.prs.C(:,k).^2)*sum(ddk_sig_hprs{k},1));
    grad_lik_gg3(idxhprs{k},:) = permute(mtimesx(mtimesx((m.prs.C(:,k)./m.prs.psi)',-permute(mu_h,[2 1 3]) + m.Y),ddk_mu_hprs{k}),[2 3 1]);
end

grad = sum(grad_lik_gg2 + grad_lik_gg3,2);