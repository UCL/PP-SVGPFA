function grad = gradLik_VariationalPrs_Poisson_svGPFA(m,Kmats,prs,idx,idx_sig,idx_sigdiag,trEval,varargin);

prs = reshape(prs,[],1,length(trEval)); % reshape into number of variational prs x number of trials
nvprs = size(prs,1);
grad_lik_poi1 = zeros(nvprs,length(trEval));
grad_lik_poi2 = zeros(nvprs,length(trEval));

mu_h = varargin{1};
var_h = varargin{2};
mask = permute(repmat(m.mask(:,trEval),[1 1 m.dy]),[1 3 2]); % check this
intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)
intval(mask) = 0;    

for k = 1:m.dx
    
    q_sqrt_k = reshape(prs(idx_sig{k},:,:),m.numZ(k),m.opts.varRnk(k),length(trEval));
    q_diag_k = prs(idx_sigdiag{k},:,:);
    
    Ktz = Kmats.Ktz{k};
    Kzzi = Kmats.Kzzi{k};
    
    Ak = mtimesx(Kzzi,Ktz,'T');
    
    grad_lik_poi1(idx{k},:) = permute(m.BinWidth*sum(mtimesx(Ak,mtimesx(intval,m.prs.C(:,k))),2),[1 3 2]);
    grad_lik_poi2(idx{k},:) = permute(sum(mtimesx(Ak,mtimesx(m.prs.C(:,k)',m.Y(:,:,trEval)),'T'),2),[1 3 2]);

    Bt = m.BinWidth*mtimesx(intval,m.prs.C(:,k).^2); % T x 1 x ntr 
    KztOut = bsxfun(@times,permute(Ktz,[2 4 1 3]),permute(Ktz,[4 2 1 3]));
    tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
    
    grad_lik_poi1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,q_sqrt_k)),[],1,length(trEval)),[1 3 2]);
    grad_lik_poi1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,diag3D(q_diag_k)))),[1 3 2]);
    
end

grad = - grad_lik_poi1(:) + grad_lik_poi2(:);


