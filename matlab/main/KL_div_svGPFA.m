function kldiv = KL_div_svGPFA(m,Kmats,ntr,varargin);

if nargin < 3 % evalutate over all trials
    trEval = 1:m.ntr;
    saveIdx = 1:m.ntr; 
    trIdx = 1:m.ntr;
    q_mu = m.q_mu;
    q_sigma = m.q_sigma;
elseif nargin == 3
    trEval = ntr;
    saveIdx(ntr) = 1; % hacky but need to keep track of indexing for all trial or single trial versions
    trIdx = 1:m.ntr;
    q_mu = m.q_mu;
    q_sigma = m.q_sigma;
else
    trEval = ntr;
    if length(trEval) == 1     
        saveIdx(ntr) = 1;
        trIdx(ntr) = 1;
    else
        saveIdx = 1:m.ntr;
        trIdx = 1:m.ntr;
    end
    q_mu = varargin{1};
    q_sigma = varargin{2};
end


kldiv = zeros(length(trEval),1);
    
for k = 1:m.dx

    Eqq = q_sigma{k} + mtimesx(q_mu{k},q_mu{k},'T');
    
    for nn = trEval
        
        KL_k = 0.5*(-logdet(Kmats.Kzzi{k}(:,:,saveIdx(nn))) - logdet(q_sigma{k}(:,:,trIdx(nn))) + ...
            vec(Kmats.Kzzi{k}(:,:,saveIdx(nn)))'*vec(Eqq(:,:,trIdx(nn)))...
            - m.numZ(k));
        
        if ~isreal(KL_k)
            error('complex value encountered')
        end
        
        kldiv(saveIdx(nn)) = kldiv(saveIdx(nn)) + KL_k;
    end
    
end

kldiv = sum(kldiv);