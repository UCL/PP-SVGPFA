function m = inducingPointMstep(m)

if ~m.fixed.Z
    if m.ntr > 1
        parfor nn = 1:m.ntr
            prs{nn} = inducingPointMstep_singleTrial(m,nn);
        end
    else
        prs = inducingPointMstep_singleTrial(m,1);
        prs = {prs};
    end
    
    % update values in model
    m = updateParameters(m,prs,4);
end

end