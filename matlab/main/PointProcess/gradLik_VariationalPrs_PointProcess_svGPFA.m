function grad = gradLik_VariationalPrs_PointProcess_svGPFA(m,Kmats,prs,idx,idx_sig,idx_sigdiag,trEval,varargin);

prs = reshape(prs,[],1,length(trEval)); % reshape into number of variational prs x number of trials
nvprs = size(prs,1);
grad_lik_pp1 = zeros(nvprs,length(trEval));
grad_lik_pp2 = zeros(nvprs,length(trEval));


if length(trEval) > 1 % evalutate over all trials
    timeEval = trEval;
elseif length(trEval) == 1
    timeEval = 1;
end


mu_h = varargin{1};
mu_hQuad = mu_h{1,1};
var_h = varargin{2};

intval = exp(mu_hQuad + 0.5*var_h); % numQuad x N x length(trEval)

for k = 1:m.dx

    q_sqrt_k = reshape(prs(idx_sig{k},:,:),m.numZ(k),m.opts.varRnk(k),length(trEval));
    q_diag_k = prs(idx_sigdiag{k},:,:);
    
    KtzQuad = Kmats.Quad.Ktz{k};
    
    Kzzi = Kmats.Kzzi{k};
    
    AkQuad = mtimesx(Kzzi,KtzQuad,'T');

    grad_lik_pp1(idx{k},:) = mtimesx(bsxfun(@times,permute(mtimesx(intval,m.prs.C(:,k)),[2 1 3]),AkQuad),m.wwQuad(:,:,trEval));
    
    for nn = 1:length(trEval);
        KtzObs = Kmats.Obs.Ktz{k,timeEval(nn)};
        AkObs = Kzzi(:,:,timeEval(nn))*KtzObs';
        grad_lik_pp2(idx{k},nn) = AkObs*m.prs.C(m.index{trEval(nn)},k);
    end                           
    
    Bt = m.wwQuad(:,:,trEval).*mtimesx(intval,(m.prs.C(:,k).^2)); % Tx1 vector
    KztOut = bsxfun(@times,permute(KtzQuad,[2 4 1 3]),permute(KtzQuad,[4 2 1 3]));
    tSum = permute(sum(bsxfun(@times, KztOut, permute(Bt, [4 2 1 3])),3),[1 2 4 3]);
    
    grad_lik_pp1(idx_sig{k},:) = permute(reshape(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,q_sqrt_k)),[],1,length(trEval)),[1 3 2]);
    grad_lik_pp1(idx_sigdiag{k},:) = permute(diag3D(mtimesx(mtimesx(Kzzi,tSum),mtimesx(Kzzi,diag3D(q_diag_k)))),[1 3 2]);
    
end
    
        
grad = - grad_lik_pp1(:) + grad_lik_pp2(:);

