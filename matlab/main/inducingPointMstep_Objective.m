function [obj, grad] = inducingPointMstep_Objective(m,prs,ntr)

% extract inducing points and hyper parameter values
Z = cell(m.dx,1);
idxZ = cell(m.dx,1);

istrt = [1 cumsum(m.num_inducing(1:end-1))+1];
iend  = cumsum(m.num_inducing);

for kk = 1:m.dx
    Z{kk} = prs(istrt(kk):iend(kk));
    idxZ{kk} = (istrt(kk):iend(kk));
end

for k = 1:m.dx
   Kzz{k} = m.kerns{k}.K(m.kerns{k}.hprs,Z{k}) + m.epsilon*eye(m.num_inducing(k));
   Kzzi{k} = pinv(Kzz{k});
   Ktz{k} = m.kerns{k}.K(m.kerns{k}.hprs,m.tt,Z{k});
   dKtzin{k} = m.kerns{k}.dKin(m.kerns{k}.hprs,m.tt,Z{k});
   [Ktt{k},~] = m.kerns{k}.Kdiag(m.kerns{k}.hprs,m.tt);
   [~,dKzzin{k}] = m.kerns{k}.dKin(m.kerns{k}.hprs,Z{k});
   allSpikes = cellvec(m.Y(:,ntr));
   KtzObsAll{k} = m.kerns{k}.K(m.kerns{k}.hprs,allSpikes,Z{k});
   dKtzObsinAll{k} = m.kerns{k}.dKin(m.kerns{k}.hprs,allSpikes,Z{k});
   
end

kld = 0;
kl_grad  = zeros(size(prs));
neuronIndex = m.neuronIndex{ntr};

% KL Divergence

for k = 1:m.dx
    kl_grad_Z = [];
    % iterate over latent processes
    
    q_mu_k = m.q_mu{k}(:,:,ntr);
    q_sqrt_k = reshape(m.q_sqrt{k}(:,:,ntr),m.num_inducing(k),m.rnk(k));

    q_diag_k = m.q_diag{k}(:,:,ntr);
    q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);
    
     KL_k = 0.5*(-logdet(Kzzi{k}) - logdet(q_sigma_k) + trace(Kzzi{k}*q_sigma_k)...
            + q_mu_k'*Kzzi{k}*q_mu_k - m.num_inducing(k));   
    kld = kld + KL_k;
    
    dKzzi = -0.5*Kzz{k} + 0.5*q_sigma_k + 0.5*(q_mu_k*q_mu_k');
    
    for ii = 1: m.num_inducing(k)
        kl_grad_Z = [kl_grad_Z;vec(dKzzi)'*vec(-(Kzzi{k}*(dKzzin{k}(:,:,ii)*Kzzi{k})))];
    end

    kl_grad(idxZ{k}) = kl_grad_Z;
    
end

% compute likelihood terms and gradient
% 
t2_grad = zeros(size(prs));
t1_grad = zeros(size(prs));

mu_k = zeros(length(m.tt),m.dx);
var_k = zeros(length(m.tt),m.dx);

% get mean and variances, gradients of likelihood
for k = 1:m.dx
    
    q_mu_k = m.q_mu{k}(:,:,ntr);
    q_sqrt_k = reshape(m.q_sqrt{k}(:,:,ntr),m.num_inducing(k),m.rnk(k));
    
    q_diag_k = m.q_diag{k}(:,:,ntr);
    q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);
    
    Ak = (Kzzi{k}*q_mu_k);
    
    mu_k(:,k) = Ktz{k}*Ak;
    Bk = Kzzi{k}*(Ktz{k}');
    
    var_k(:,k) = (Ktt{k} + diag(Bk'*(q_sigma_k - Kzz{k})*Bk));
    
    % evaluate mean function at observed spikes
    muObsAll(:,k) = KtzObsAll{k}*Ak;
    
end

[mu_h, var_h] = map_to_nontransformed_rate(m,mu_k,var_k);

if strcmp(m.nonlin.name,'Exponential')
    % value of integral over intensity function
    intval = exp(mu_h + 0.5*var_h); % T x N
    % naive quadrature for evaluating integral
    t1 = m.T/m.ng*sum(intval(:));
    
    % evaulate second term of likelihood with observed event times

    mu_sumAll = sum(indexed_nontransformed_rate(m,neuronIndex,muObsAll));
    t2 = mu_sumAll;
    
    for k = 1:m.dx

        q_mu_k = m.q_mu{k}(:,:,ntr);
        q_sqrt_k = reshape(m.q_sqrt{k}(:,:,ntr),m.num_inducing(k),m.rnk(k));
        
        q_diag_k = m.q_diag{k}(:,:,ntr);
        q_sigma_k = q_sqrt_k*q_sqrt_k' + diag(q_diag_k.^2);
        
        Ak = (Kzzi{k}*q_mu_k);
        mu_k(:,k) = Ktz{k}*Ak;
                
        
        dBzz = -(Kzzi{k}*reshape(permute(reshape((Kzzi{k}*dKzzin{k}(:,:)),...
            m.num_inducing(k),m.num_inducing(k),[]),[2 1 3]),m.num_inducing(k), m.num_inducing(k)^2,[]));
        
        dBzz = reshape(dBzz,m.num_inducing(k),m.num_inducing(k),[]);
       
        ddObsin1 = permute(mtimesx(Ak,'T',permute(dKtzObsinAll{k},[2 1 3])),[2 3 1]);
        ddObsin2 = permute(mtimesx(mtimesx(KtzObsAll{k},dBzz),q_mu_k),[1 3 2]);
        ggn2 = m.C(neuronIndex,k)'*(ddObsin1 + ddObsin2);
       
        t2_grad(idxZ{k}) = ggn2;
      
        mm1 = reshape(Ak'*reshape(permute(dKtzin{k},[2 1 3]),m.num_inducing(k),[],1),[],m.num_inducing(k));
        mm2 = mtimesx(q_mu_k,permute(Ktz{k},[3 2 1]));
        mm22 = mtimesx(reshape(mm2,[],m.ng)',reshape(permute(dBzz,[2 1 4 3]),[],1,m.num_inducing(k)));
        ddk_mu_in = mm1 + permute(mm22,[1 3 2]);
     
        mm4 = (Kzzi{k}*Ktz{k}');
        mm3 = q_sigma_k*mm4;
        mm5 = bsxfun(@times,permute(mm3,[1 3 2]),permute(Ktz{k},[3 2 1]));
        mm5 = reshape(mm5,[],1,m.ng);
        mm7 = Kzzi{k}*mm3;
        mm6 = permute(reshape(dBzz,[],m.num_inducing(k)),[1 2 3]); 
        
        ddk_sig_in = 2*permute(mtimesx(permute(mm7,[3 1 2]),permute(dKtzin{k},[3 2 1])),[3 2 1]) ...
            + 2*permute(mtimesx(mm6',mm5),[3 1 2]) ...
            - 2*permute(mtimesx(permute(dKtzin{k},[2 3 1]),permute(mm4,[1 3 2])),[3 1 2]) ...
            - permute(mtimesx(permute(Ktz{k},[3 2 1]),permute(mtimesx(permute(dBzz,[1 2 4 3]),Ktz{k},'T'),[1 4 2 3])),[3 2 1]);
        
        mm8 = bsxfun(@times,permute(ddk_mu_in,[1 3 2]),m.C(:,k)') + 0.5*bsxfun(@times,permute(ddk_sig_in,[1 3 2]),m.C(:,k).^2');
        
        t1_grad_Z = m.T/m.ng*mtimesx(intval(:)',reshape(mm8,[],m.num_inducing(k)))';

        t1_grad(idxZ{k}) = t1_grad_Z;
        
    end
    
else
    error('specified non-linearity not implemented')
end


% add terms together and return obj and grad values
obj =  t1 - t2 + kld; 
grad = t1_grad - t2_grad + kl_grad;
