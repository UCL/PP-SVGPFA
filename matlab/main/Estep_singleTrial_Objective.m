function [obj,grad] = Estep_singleTrial_Objective(m,prs,Kmats,ntr)

% get indices to extract variational parameters from prs vector
idx = cell(m.dx,1);
idx_sig = cell(m.dx,1);
idx_sigdiag = cell(m.dx,1);
q_mu = cell(m.dx,1);
q_sqrt = cell(m.dx,1);
q_diag = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

num_cov = m.numZ.*m.opts.varRnk;
istrt_sig = iend(end) + [1 cumsum(num_cov(1:end-1))+1];
iend_sig  = iend(end) + cumsum(num_cov);

istrt_sigdiag = iend_sig(end) + [1 cumsum(m.numZ(1:end-1))+1];
iend_sigdiag = iend_sig(end) + cumsum(m.numZ);

for k = 1:m.dx
    idx{k} = istrt(k):iend(k);
    idx_sig{k} = istrt_sig(k):iend_sig(k);
    idx_sigdiag{k} = istrt_sigdiag(k):iend_sigdiag(k);
    
    q_mu{k} = prs(idx{k});
    q_sqrt{k} = reshape(prs(idx_sig{k}),m.numZ(k),m.opts.varRnk(k));
    q_diag{k} = prs(idx_sigdiag{k});
end


for k = 1:m.dx
    idx{k} = istrt(k):iend(k);
    idx_sig{k} = istrt_sig(k):iend_sig(k);
    idx_sigdiag{k} = istrt_sigdiag(k):iend_sigdiag(k);
    
    q_mu{k} = prs(idx{k});
    q_sqrt{k} = reshape(prs(idx_sig{k}),m.numZ(k),m.opts.varRnk(k));
    q_diag{k} = prs(idx_sigdiag{k});
    
    q_sigma{k} = mtimesx(q_sqrt{k} ,q_sqrt{k},'T') + diag3D(q_diag{k}.^2);
        
    Ak = mtimesx(Kmats.Kzzi{k},q_mu{k});
    
    if isfield(Kmats,'Quad') % need mean and variance for quad, mean for observed spikes
        Bkf = mtimesx(Kmats.Kzzi{k},Kmats.Quad.Ktz{k},'T');
        mm1f = mtimesx( q_sigma{k} - Kmats.Kzz{k},Bkf);
        mu_k{1,1}(:,k) = mtimesx(Kmats.Quad.Ktz{k},Ak);
        mu_k{2,1}(:,k) = mtimesx(Kmats.Obs.Ktz{k},Ak);
        var_k(:,k) = bsxfun(@plus,Kmats.Quad.Ktt(:,k), sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));
    else
        Bkf = mtimesx(Kmats.Kzzi{k},Kmats.Ktz{k},'T');
        
        mm1f = mtimesx(q_sigma{k} - Kmats.Kzz{k},Bkf);
        mu_k(:,k) = mtimesx(Kmats.Ktz{k},Ak);
        var_k(:,k) = bsxfun(@plus,Kmats.Ktt(:,k), sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));
    end
end

if iscell(mu_k) % applies to point process version
    mu_h{1,1} = bsxfun(@plus,mtimesx(mu_k{1},m.prs.C'), m.prs.b');
    mu_h{2,1} = bsxfun(@plus,sum(bsxfun(@times,mu_k{2,1},m.prs.C(m.index{ntr},:)),2), m.prs.b(m.index{ntr}));
else
    mu_h = bsxfun(@plus,mtimesx(mu_k,m.prs.C'), m.prs.b');
end
var_h = mtimesx(var_k,(m.prs.C.^2)');
% 

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h,ntr);
gradElik = m.EMfunctions.gradLik_variationalPrs(m,Kmats,prs,idx,idx_sig,idx_sigdiag,ntr,mu_h,var_h);

% get KL divergence and gradient
KLd = KL_div_svGPFA(m,Kmats,ntr,q_mu,q_sigma);
gradKLd = KLgrad_variationalPrs_svGPFA(m,Kmats,q_mu,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,ntr);

obj = - Elik + KLd; % negative free energy
grad = - gradElik + gradKLd; % gradients


