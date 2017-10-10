function kl_grad = KLgrad_inducingPoints_svGPFA(m,Kmats,idxZ,trEval);
% currently only for a single trial assuming Z always updates separately

kl_grad  = zeros(sum(m.numZ),length(trEval));

% KL Divergence gradients

for k = 1:m.dx
    
    kl_grad_Z = [];
    % iterate over latent processes
    
    q_mu_k = m.q_mu{k}(:,:,trEval);
    q_sigma_k = m.q_sigma{k}(:,:,trEval);
    
    Eqq = q_sigma_k + mtimesx(q_mu_k,q_mu_k,'T');
    dKzzi = -0.5*Kmats.Kzz{k} + 0.5*(Eqq);
    
    for ii = 1: m.numZ(k)
        kl_grad_Z = [kl_grad_Z;...
            mtimesx(reshape(dKzzi,[],1,length(trEval)),'T',...
            reshape(-mtimesx(Kmats.Kzzi{k},mtimesx(permute(Kmats.dKzzin{k}(:,:,ii,:),[1 2 4 3]),Kmats.Kzzi{k})),[],1,length(trEval)))];
%          vec(dKzzi)'*vec(-(Kmats.Kzzi{k}*(Kmats.dKzzin{k}(:,:,ii)*Kmats.Kzzi{k})))];
    end

    kl_grad(idxZ{k},:) = permute(kl_grad_Z,[1 3 2]);
    
end

kl_grad = kl_grad(:);
