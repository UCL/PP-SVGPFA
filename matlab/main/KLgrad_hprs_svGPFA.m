function kl_grad = KLgrad_hprs_svGPFA(m,Kmats,idxhprs);
% gradient of KL diveregence wrt hyperparameters, computed for all trials
% simultanously.

for k = 1:m.dx
    kl_grad_hprs = [];
    
    Eqq = m.q_sigma{k} + mtimesx(m.q_mu{k},m.q_mu{k},'T');
    dKzzi = -0.5*Kmats.Kzz{k} + 0.5*(Eqq);
    
    for hh = 1: m.kerns{k}.numhprs
        kl_grad_hprs = [kl_grad_hprs;...
            mtimesx(reshape(dKzzi,[],1,m.ntr),'T',...
            reshape(-mtimesx(Kmats.Kzzi{k},mtimesx(permute(Kmats.dKzzhprs{k}(:,:,hh,:),[1 2 4 3]),Kmats.Kzzi{k})),[],1,m.ntr))];
    end
    
    kl_grad(idxhprs{k},:) = permute(kl_grad_hprs,[1 3 2]);
end

kl_grad = sum(kl_grad,2); % sum over all trials