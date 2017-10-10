function grad = gradLik_inducingPoints_Gaussian_svGPFA(m,Kmats,idxZ,trEval,varargin);

% trEval is a vector with 1:m.ntr or just a single tria index

grad_lik_gg2 = zeros(sum(m.numZ),length(trEval));
grad_lik_gg3 = zeros(sum(m.numZ),length(trEval));

mu_h = varargin{1};
mask = permute(repmat(m.mask(:,trEval),[1 1 m.dy]),[1 3 2]); % check this
mu_h(mask) = 0;

R = max(m.trLen);

[ddk_mu_in,ddk_sig_in] = grads_inducingPoints_posteriorGP_svGPFA(m,Kmats,trEval,R);

for k = 1:m.dx
    grad_lik_gg2(idxZ{k},:) = -0.5*squeeze((1./m.prs.psi'*m.prs.C(:,k).^2)*sum(ddk_sig_in{k},1));    
    grad_lik_gg3(idxZ{k},:) = permute(mtimesx(mtimesx((m.prs.C(:,k)./m.prs.psi)',-permute(mu_h,[2 1 3]) + m.Y(:,:,trEval)),ddk_mu_in{k}),[2 3 1]);
end

grad = grad_lik_gg2(:) + grad_lik_gg3(:);