function [obj, grad] = hyperMstep_Objective(m,prs)
% extract hyperparameter values
%%
hprs = cell(m.dx,1);
idxhprs = cell(m.dx,1);

hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'UniformOutput',false)'));
istrthprs = [1; hprsidx(1:end-1)+1];
iendhprs = hprsidx;

for kk = 1:m.dx
    hprs{kk} = prs(istrthprs(kk):iendhprs(kk));
    idxhprs{kk} = (istrthprs(kk):iendhprs(kk));
end

% pre-allocate cell array for saving kernel matrices and gradients

Kzz = cell(m.dx,1);
Kzzi = cell(m.dx,1);
dKzzhprs = cell(m.dx,1);
Ktz = cell(m.dx,1);
dKtzhprs = cell(m.dx,1);
Ktt = zeros(m.ng,m.dx);
dKtt = cell(m.dx,1);
allSpikes = cell(m.ntr,1);
KtzObsAll = cell(m.dx,m.ntr);
dKtzObshprsAll = cell(m.dx,m.ntr);
neuronIndex = m.neuronIndex;
% build kernel matrices
for k = 1:m.dx 
    
    [Ktt(:,k),dKtt{k}] = m.kerns{k}.Kdiag(hprs{k},m.tt);
    
    for numtr = 1:m.ntr
        Kzz{k}(:,:,numtr) = m.kerns{k}.K(hprs{k},m.Z{k,numtr}) + m.epsilon*eye(m.num_inducing(k));
        Kzzi{k}(:,:,numtr) = pinv(Kzz{k}(:,:,numtr));

        dKzzhprs{k}(:,:,:,numtr) = m.kerns{k}.dKhprs(hprs{k},m.Z{k,numtr});
        
        Ktz{k}(:,:,numtr) = m.kerns{k}.K(hprs{k},m.tt,m.Z{k,numtr});
        
        dKtzhprs{k}(:,:,:,numtr) = m.kerns{k}.dKhprs(hprs{k},m.tt,m.Z{k,numtr});
        
        allSpikes{numtr} = cellvec(m.Y(:,numtr));
        KtzObsAll{k,numtr} = m.kerns{k}.K(hprs{k},allSpikes{numtr},m.Z{k,numtr});
        dKtzObshprsAll{k,numtr} = m.kerns{k}.dKhprs(hprs{k},allSpikes{numtr},m.Z{k,numtr});
        
    end   
end
%%

kld = zeros(m.ntr,1);
kl_grad  = zeros(size(prs,1),m.ntr);
t2_grad = zeros(size(prs,1),m.ntr);
t1_grad = zeros(size(prs,1),m.ntr);
t2 = zeros(m.ntr,1);


mu_k = zeros(m.ng,m.dx,m.ntr);
var_k = zeros(m.ng,m.dx,m.ntr);

q_mu_k = cell(m.dx,1);
q_sigma_k = cell(m.dx,1);
Eqq = cell(m.dx,1);
Bk = cell(m.dx,1);

% get mean and variances, rates etc. for all trials
for k = 1:m.dx 
    
    q_mu_k{k} = m.q_mu{k}; % now 3D over trials
    q_sqrt_k = reshape(m.q_sqrt{k},m.num_inducing(k),m.rnk(k),m.ntr);
    q_diag_k = diag3D(m.q_diag{k});
    q_sigma_k{k} = mtimesx(q_sqrt_k,q_sqrt_k,'T') + (q_diag_k.^2);
    
    Eqq{k} = q_sigma_k{k} + mtimesx(q_mu_k{k},q_mu_k{k},'T');
    
    Ak{k} = mtimesx(Kzzi{k},q_mu_k{k});
    Bk{k} = mtimesx(Kzzi{k},Ktz{k},'T');
    
    mu_k(:,k,:) = mtimesx(Ktz{k},Ak{k});
    mm1 = mtimesx(q_sigma_k{k} - Kzz{k},Bk{k});
    
    var_k(:,k,:) = bsxfun(@plus,Ktt(:,k), sum(bsxfun(@times,permute(Bk{k},[2 1 3]),permute(mm1,[2 1 3])),2));


    for numtr = 1:m.ntr
        muObsAll{numtr}(:,k) = KtzObsAll{k,numtr}*Ak{k}(:,:,numtr);
    end

end
[mu_h, var_h] = map_to_nontransformed_rate(m,mu_k,var_k);




%% calculate likelihood terms
intval = m.nonlin.Exponential(mu_h + 0.5*var_h);
t1 = permute(m.T/m.ng*sum(reshape(intval,[m.ng*m.dy 1 m.ntr]),1),[3 2 1]);

for numtr = 1:m.ntr
    
    mu_sumAll = sum(indexed_nontransformed_rate(m,neuronIndex{numtr},muObsAll{numtr}));
    t2(numtr) = mu_sumAll;

end

%% KL-divergence term
for k = 1:m.dx
    
    dKzzi = -0.5*Kzz{k} + 0.5*(Eqq{k});
    
    for numtr = 1:m.ntr
        
        % KL Divergence for each trial
        kl_grad_hprs = [];
        % iterate over latent processes
        
        KL_k = 0.5*(-logdet(Kzzi{k}(:,:,numtr)) - logdet(q_sigma_k{k}(:,:,numtr)) + ...
            vec(Kzzi{k}(:,:,numtr))'*vec(Eqq{k}(:,:,numtr))...
            - m.num_inducing(k));
        
        
        if ~isreal(KL_k)
            error('complex val encountered')
        end
        
        kld(numtr) = kld(numtr) + KL_k;
    end
    
    
    for hh = 1: m.kerns{k}.numhprs
        
        kl_grad_hprs = [kl_grad_hprs;...
            mtimesx(reshape(dKzzi,[],1,m.ntr),'T',...
            reshape(-mtimesx(Kzzi{k},mtimesx(permute(dKzzhprs{k}(:,:,hh,:),[1 2 4 3]),Kzzi{k})),[],1,m.ntr))];
    end
    
    kl_grad(idxhprs{k},:) = permute(kl_grad_hprs,[1 3 2]);
    
    % compute likelihood terms and gradient for each trial
    
end

%% calculate gradients

for k = 1:m.dx
    
    dBhh = -mtimesx(Kzzi{k},...
        reshape(permute(reshape(...
        mtimesx(Kzzi{k},...
        reshape(dKzzhprs{k},m.num_inducing(k),[],m.ntr)),...
        m.num_inducing(k),m.num_inducing(k),[]),[2 1 3]),...
        m.num_inducing(k), m.kerns{k}.numhprs*m.num_inducing(k),[]));
    
    dBhh = reshape(dBhh,m.num_inducing(k),m.num_inducing(k),[],m.ntr);
    
    
    if m.ntr == 1
        for numtr = 1:m.ntr % loop over trials since inducing point locations may be different

             ddObshprs1 = permute(mtimesx(Ak{k}(:,:,numtr),'T',permute(dKtzObshprsAll{k,numtr},[2 1 3])),[2 3 1]);
             ddObshprs2 = permute(mtimesx(mtimesx(KtzObsAll{k,numtr},dBhh(:,:,:,numtr)),q_mu_k{k}(:,:,numtr)),[1 3 2]);           
             ggn{numtr} = m.C(neuronIndex{numtr},k)'*(ddObshprs1 + ddObshprs2);
        end
        
    else
        parfor numtr = 1:m.ntr % do loop in parallel when more than one trial 

            ddObshprs1 = permute(mtimesx(Ak{k}(:,:,numtr),'T',permute(dKtzObshprsAll{k,numtr},[2 1 3])),[2 3 1]);
            ddObshprs2 = permute(mtimesx(mtimesx(KtzObsAll{k,numtr},dBhh(:,:,:,numtr)),q_mu_k{k}(:,:,numtr)),[1 3 2]);
            ggn{numtr} = m.C(neuronIndex{numtr},k)'*(ddObshprs1 + ddObshprs2);
            
        end
    end
    t2_grad(idxhprs{k},:) = cell2mat(ggn')';
    
    mm1 = reshape(...
        mtimesx(Ak{k},'T',...
        reshape(permute(dKtzhprs{k},[2 1 3 4]),m.num_inducing(k),[],m.ntr)),[],m.kerns{k}.numhprs,m.ntr);
    
    mm2 = mtimesx(permute(q_mu_k{k},[1 2 4 3]),permute(Ktz{k},[4 2 1 3]));
    
    mm22 = mtimesx(reshape(mm2,[],m.ng,1,m.ntr),'T',reshape(permute(dBhh,[2 1 5 3 4]),[],1,m.kerns{k}.numhprs,m.ntr));
    ddk_mu_hprs = mm1 + squeeze(mm22);
    
    mm4 = mtimesx(Kzzi{k},Ktz{k},'T');
    mm3 = mtimesx(q_sigma_k{k},mm4);
    mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Ktz{k},[4 2 1 3]));
    mm5 = reshape(mm5,[],1,m.ng,m.ntr);
    mm6 = permute(reshape(dBhh,[],m.kerns{k}.numhprs,m.ntr),[2 1 4 3]);
    mm7 = mtimesx(Kzzi{k},mm3);
    
    ttm1 = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(dKtzhprs{k},[2 3 1 4])),[3 2 4 1]) ...
        + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
        - 2*permute(mtimesx(permute(dKtzhprs{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
        - permute(mtimesx(permute(Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBhh,[1 2 4 3]),Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
    ddk_sig_hprs = bsxfun(@plus,permute(dKtt{k},[1 3 2]),ttm1);
    
    mm8 = bsxfun(@times,permute(ddk_mu_hprs,[1 5 2 3 4]),m.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_hprs,[1 5 2 3 4]),m.C(:,k).^2');
    
    t1_grad(idxhprs{k},:) = permute(m.T/m.ng*mtimesx(reshape(intval,[],1,m.ntr),'T',reshape(mm8,[],m.kerns{k}.numhprs,m.ntr)),[2 3 1]);
      
    
end
%%
% add terms together and return obj and grad values
obj  = sum(t1 - t2 + kld); % sum over trials
grad = sum(t1_grad - t2_grad + kl_grad,2);
