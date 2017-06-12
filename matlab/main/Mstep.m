function m = Mstep(m);

% function to carry out M step optimization to learn C and b given
% everything else

if ~ m.fixed.prs
    
    % extract M step parameters to be optimized
    C0 = m.C;
    b0 = m.b;
    prs0 = [C0(:);b0];
    
    Kzz = cell(m.dx,1);
    Kzzi = cell(m.dx,1);
    Ktz = cell(m.dx,1);
    KtzObsAll = cell(m.dx,m.ntr);
    Ktt = zeros(m.ng,m.dx);
    mu_k = zeros(m.ng,m.dx,m.ntr);
    var_k = zeros(m.ng,m.dx,m.ntr);
    muObsAll = cell(m.ntr);
    for k = 1:m.dx
        Ktt(:,k) = m.kerns{k}.Kdiag(m.kerns{k}.hprs,m.tt);
        for numtr = 1:m.ntr
            Kzz{k}(:,:,numtr) = m.kerns{k}.K(m.kerns{k}.hprs,m.Z{k,numtr}) + m.epsilon*eye(m.num_inducing(k));
            Kzzi{k}(:,:,numtr) = pinv(Kzz{k}(:,:,numtr));
            Ktz{k}(:,:,numtr) = m.kerns{k}.K(m.kerns{k}.hprs,m.tt,m.Z{k,numtr});
            KtzObsAll{k,numtr} = m.kerns{k}.K(m.kerns{k}.hprs,cellvec(m.Y(:,numtr)),m.Z{k,numtr});
        end
    end
    
    
    for k = 1:m.dx
        
        q_mu_k = m.q_mu{k}; % now 3D over trials
        q_sqrt_k = reshape(m.q_sqrt{k},m.num_inducing(k),m.rnk(k),m.ntr);
        q_diag_k = diag3D(m.q_diag{k});
        q_sigma_k = mtimesx(q_sqrt_k,q_sqrt_k,'T') + (q_diag_k.^2);
                
        Ak = mtimesx(Kzzi{k},q_mu_k);
        Bk = mtimesx(Kzzi{k},Ktz{k},'T');
        
        mu_k(:,k,:) = mtimesx(Ktz{k},Ak);
        mm1 = mtimesx(q_sigma_k - Kzz{k},Bk);
        
        var_k(:,k,:) = bsxfun(@plus,Ktt(:,k), sum(bsxfun(@times,permute(Bk,[2 1 3]),permute(mm1,[2 1 3])),2));
        
        
        for numtr = 1:m.ntr
            muObsAll{numtr}(:,k) = KtzObsAll{k,numtr}*Ak(:,:,numtr);
        end
        
    end
   
    % make objective function handle
    fun = @(prs) Mstep_Objective(m,prs,mu_k,var_k,muObsAll);
    opts = optimset('Gradobj','on','display', 'none');
    opts.MaxIter = 100;
    prs = minFunc(fun,prs0,opts);
    % update values in model
    m = updateParameters(m,prs,2);
    
end

end