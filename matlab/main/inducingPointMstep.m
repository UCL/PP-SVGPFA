function m = inducingPointMstep(m)

if ~m.opts.fixed.Z
    
    istrt = [1 cumsum(m.numZ(1:end-1))+1];
    iend  = cumsum(m.numZ);
    
    if m.opts.parallel
        parfor (nn = 1:m.ntr,m.opts.numWorkers)
            prs{nn} = inducingPointMstep_singleTrial(m,nn);
        end
        
        % update inducing point values in model
        for nn = 1:m.ntr
            for kk = 1:m.dx
                m.Z{kk}(:,:,nn) = prs{nn}(istrt(kk):iend(kk));
            end
        end
        
    else
        
        prs = inducingPointMstep_allTrials(m);
        
        % update inducing point values in model
        prs = reshape(prs,[],1, m.ntr);
        for kk = 1:m.dx
            m.Z{kk} = prs(istrt(kk):iend(kk),:,:);
        end
        
    end
    
end