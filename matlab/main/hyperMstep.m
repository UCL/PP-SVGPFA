function m = hyperMstep(m)
% function to update hyperparameters
if ~m.opts.fixed.hprs
    % extract hyperparameters for each GP kernel
    prs0 = cell2mat(cellfun(@(c)c.hprs, m.kerns,'uni',0)');
    
    % make objective function
    fun = @(prs) hyperMstep_Objective(m,prs);
%     DerivCheck(fun,prs0);
    optimopts = optimset('Gradobj','on','display', 'none');
    
    optimopts.MaxIter = m.opts.maxiter.hyperMstep;
    prs = minFunc(fun,prs0,optimopts);
    
    % update kernel hyperparameters in model structure
    hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'uni',0)'));
    istrthprs = [1; hprsidx(1:end-1)+1];
    iendhprs = hprsidx;
    
    for kk = 1:m.dx
        m.kerns{kk}.hprs = prs(istrthprs(kk):iendhprs(kk));
    end
    
end
