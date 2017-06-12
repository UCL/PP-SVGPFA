function [obj, grad] = Estep_Objective(m,prs,allKzzi,allKtz,allKtt,KtzObsAll,neuronIndex)

% get indices to extract variational parameters from prs vector
idx = cell(m.dx,1);
idx_sig = cell(m.dx,1);
idx_sigdiag = cell(m.dx,1);

istrt = [1 cumsum(m.num_inducing(1:end-1))+1];
iend  = cumsum(m.num_inducing);

num_cov = m.num_inducing.*m.rnk;
istrt_sig = iend(end) + [1 cumsum(num_cov(1:end-1))+1];
iend_sig  = iend(end) + cumsum(num_cov);

istrt_sigdiag = iend_sig(end) + [1 cumsum(m.num_inducing(1:end-1))+1];
iend_sigdiag = iend_sig(end) + cumsum(m.num_inducing);

for kk = 1:m.dx
    idx{kk} = istrt(kk):iend(kk);
    idx_sig{kk} = istrt_sig(kk):iend_sig(kk);
    idx_sigdiag{kk} = istrt_sigdiag(kk):iend_sigdiag(kk);
end

%% compute KL term and KL gradient
grad_kl = zeros(size(prs));
kld = 0;

for k = 1:m.dx % iterate over latent processes

    q_mu_k = prs(idx{k});
    q_sqrt_k = reshape(prs(idx_sig{k}),m.num_inducing(k),m.rnk(k));
    q_diag_k = prs(idx_sigdiag{k});
    q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);   
    Kzzi = allKzzi{k};

    KL_k = 0.5*(-logdet(Kzzi) - logdet(q_sigma_k) + Kzzi(:)'*q_sigma_k(:)...
        + q_mu_k'*(Kzzi*q_mu_k) - m.num_inducing(k)); % compute KL divergence for current term
    
    kld = kld + KL_k;
    
    % compute gradient of KL divergence    
    grad_kl(idx{k}) =  (Kzzi*q_mu_k);%faster
    grad_kl(idx_sig{k}) =  vec( - q_sigma_k'\q_sqrt_k + Kzzi*q_sqrt_k);
    grad_kl(idx_sigdiag{k}) =  diag( - q_sigma_k\diag(q_diag_k) +  Kzzi*diag(q_diag_k));

end

%% compute likelihood terms and gradient
grad_lik1 = zeros(size(prs));
grad_lik2 = zeros(size(prs));
mu_k= zeros(length(m.tt),m.dx);
var_k = zeros(length(m.tt),m.dx);
% get mean and variance estimate for current value of latents
for k = 1:m.dx
    
    q_mu_k = prs(idx{k});
    q_sqrt_k = reshape(prs(idx_sig{k}),m.num_inducing(k),m.rnk(k));

    q_diag_k = prs(idx_sigdiag{k});
    q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);
    
    Kzzi = allKzzi{k};
    Ktz = allKtz{k};
    Ktt = allKtt{k};

    Ak = Kzzi*Ktz';
    mu_k(:,k) = Ak'*q_mu_k;
    var_k(:,k) = (Ktt +  sum((Ak'*(q_sigma_k - pinv(Kzzi))).*Ak',2));
        
    muObsAll(:,k) = KtzObsAll{k}*(Kzzi*q_mu_k);
    
end

[mu_h, var_h] = map_to_nontransformed_rate(m,mu_k,var_k);
    
if strcmp(m.nonlin.name,'Exponential')
    % value of integral over intensity function
    intval = exp(mu_h + 0.5*var_h);  % T x N 
    % naive quadrature for evaluating integral
    t1 = m.T/m.ng*sum(intval(:));
    
    mu_sumAll = sum(indexed_nontransformed_rate(m,neuronIndex,muObsAll));
    t2 = mu_sumAll;
    
    for k = 1:m.dx
        
        q_sqrt_k = reshape(prs(idx_sig{k}),m.num_inducing(k),m.rnk(k));
        q_diag_k = prs(idx_sigdiag{k});

        Kzt = allKtz{k}';
        Kzzi = allKzzi{k};
        
        Ak = Kzzi*Kzt;

        grad_lik1(idx{k}) = m.T/m.ng*sum(bsxfun(@times,(intval*m.C(:,k))', Ak),2);
        
        Bt = intval*(m.C(:,k).^2); % Tx1 vector
        
        KztOut = mtimesx(permute(Kzt,[1 3 2]),permute(Kzt,[3 1 2]));
        tSum = permute(mtimesx(permute(KztOut,[2 3 1]), Bt),[1 3 2]); 
     
        grad_lik1(idx_sig{k}) = vec((m.T/m.ng*Kzzi*tSum)*(Kzzi*q_sqrt_k));
        grad_lik1(idx_sigdiag{k}) = diag(m.T/m.ng*(Kzzi*tSum)*(Kzzi*diag(q_diag_k)));
        
        grad_lik2(idx{k}) = Kzzi*(KtzObsAll{k}'*m.C(neuronIndex,k));
    end
    
else % other non-linearities
    error('specified non-linearity not implemented')
end

% add terms together and return obj and grad values
obj =  t1 - t2 + kld; 
grad = grad_lik1 - grad_lik2 + grad_kl;

end