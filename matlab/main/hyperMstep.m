function m = hyperMstep(m)
% function to update hyperparameters
if ~ m.fixed.hprs
    % extract hyperparameters for each GP kernel
    prs0 = cell2mat(cellfun(@(c)c.hprs, m.kerns,'uni',0)');
    
    % make objective function
    fun = @(prs) hyperMstep_Objective(m,prs);

    opts = optimset('Gradobj','on','display', 'none');
    
    opts.MaxIter = 15;
    prs = minFunc(fun,prs0,opts);
    
    if fun(prs) > fun(prs0)
        error('increase in objective')
    end
    
    m = updateParameters(m,prs,3);
    
end

end