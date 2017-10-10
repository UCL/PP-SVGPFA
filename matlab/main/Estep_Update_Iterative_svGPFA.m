function m = Estep_Update_Iterative_svGPFA(m,Kmats);

idx = cell(m.dx,1);
idx_sig = cell(m.dx,1);
idx_sigdiag = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

num_cov = m.numZ.*m.opts.varRnk;
istrt_sig = iend(end) + [1 cumsum(num_cov(1:end-1))+1];
iend_sig  = iend(end) + cumsum(num_cov);

istrt_sigdiag = iend_sig(end) + [1 cumsum(m.numZ(1:end-1))+1];
iend_sigdiag = iend_sig(end) + cumsum(m.numZ);

for kk = 1:m.dx
    idx{kk} = istrt(kk):iend(kk);
    idx_sig{kk} = istrt_sig(kk):iend_sig(kk);
    idx_sigdiag{kk} = istrt_sigdiag(kk):iend_sigdiag(kk);
end


if m.ntr == 1
    prs = Estep_Update_singleTrial(m,Kmats,1);
    for k = 1:m.dx
        m.q_mu{k} = prs(idx{k},:,:);
        m.q_sqrt{k} = reshape(prs(idx_sig{k},:,:),m.numZ(k),m.opts.varRnk(k),m.ntr);
        m.q_diag{k} = prs(idx_sigdiag{k},:,:);
        qq =  reshape(m.q_sqrt{k},m.numZ(k),m.opts.varRnk(k),m.ntr);
        dd = diag3D(m.q_diag{k}.^2);
        m.q_sigma{k} = mtimesx(qq,qq,'T') + dd;
    end
    
elseif m.opts.parallel % optimise variational parameters for each trial in parallel
    
    parfor (nn = 1:m.ntr,m.opts.numWorkers)
        prs{nn} = Estep_Update_singleTrial(m,Kmats,nn);
    end
    
    % update variational parameter values in model
    
    for kk = 1:m.dx
        for nn = 1:m.ntr
            m.q_mu{kk}(:,:,nn) = prs{nn}(idx{kk});
            m.q_sqrt{kk}(:,:,nn) = prs{nn}(idx_sig{kk});
            m.q_diag{kk}(:,:,nn) = prs{nn}(idx_sigdiag{kk});
        end
        qq =  reshape(m.q_sqrt{kk},m.numZ(kk),m.opts.varRnk(kk),m.ntr);
        dd = diag3D(m.q_diag{kk}.^2);
        m.q_sigma{kk} = mtimesx(qq,qq,'T') + dd;
    end
else % optimise variational parameters over all trials at once
    
    prs = Estep_Update_allTrials(m,Kmats);
    % update variational parameter values in model
    prs = reshape(prs,[],1,m.ntr);
    for k = 1:m.dx
        m.q_mu{k} = prs(idx{k},:,:);
        m.q_sqrt{k} = reshape(prs(idx_sig{k},:,:),m.numZ(k),m.opts.varRnk(k),m.ntr);
        m.q_diag{k} = prs(idx_sigdiag{k},:,:);
        qq =  reshape(m.q_sqrt{k},m.numZ(k),m.opts.varRnk(k),m.ntr);
        dd = diag3D(m.q_diag{k}.^2);
        m.q_sigma{k} = mtimesx(qq,qq,'T') + dd;
    end
end