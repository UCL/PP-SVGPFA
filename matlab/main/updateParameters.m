function m = updateParameters(m,prs,flag)

% function to update model struct with most recent parameter values

switch flag
    case 1
        idx = cell(m.dx,1);
        idx_sig = cell(m.dx,1);
        idx_sigdiag = cell(m.dx,1);
        
        istrt = [1 cumsum(m.num_inducing(1:end-1))+1];
        iend  = cumsum(m.num_inducing);
        
        num_cov = m.num_inducing.*m.rnk;
        istrt_sig = iend(end) + [1 cumsum(num_cov(1:end-1))+1];
        iend_sig  = iend(end) + cumsum(num_cov);
        
        istrt_sigdiag = iend_sig(end) + [1 cumsum(m.num_inducing(1:end-1))+1];
        iend_sigdiag = iend_sig(end) + cumsum(m.num_inducing);
        
        for kk = 1:m.dx
            idx{kk} = istrt(kk):iend(kk);
            idx_sig{kk} = istrt_sig(kk):iend_sig(kk);
            idx_sigdiag{kk} = istrt_sigdiag(kk):iend_sigdiag(kk);
        end
        
        % update m using prs for all trials
        for nn = 1:m.ntr
            for kk = 1:m.dx
                m.q_mu{kk}(:,:,nn) = prs{nn}(idx{kk});
                m.q_sqrt{kk}(:,:,nn) = prs{nn}(idx_sig{kk});
                m.q_diag{kk}(:,:,nn) = prs{nn}(idx_sigdiag{kk});
            end
        end
        
    case 2
        % update m using prs
        m.C = reshape(prs(1:m.dy*m.dx),[m.dy, m.dx]);
        m.b = prs(m.dy*m.dx + 1 : end);
        
    case 3
        % extract kernel hyper parameters and inducing points
        % from prs vector and update kerns structure.
        hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'UniformOutput',false)'));
        istrthprs = [1; hprsidx(1:end-1)+1];
        iendhprs = hprsidx;
        
        for kk = 1:m.dx
            m.kerns{kk}.hprs = prs(istrthprs(kk):iendhprs(kk));
        end
        
    case 4
        % extract inducing points from prs vector and update m
        istrt = [1 cumsum(m.num_inducing(1:end-1))+1];
        iend  = cumsum(m.num_inducing);
       
        for nn = 1:m.ntr
            for kk = 1:m.dx
                m.Z{kk,nn} = prs{nn}(istrt(kk):iend(kk));
            end
        end
end