function opts = setOptionValues_svGPFA(m,options);

opts = options;

if isfield(opts,'fixed')
    if ~isfield(opts.fixed,'Z')
        opts.fixed.Z = 0;
    end
    
    if ~isfield(opts.fixed,'prs')
        opts.fixed.prs = 0;
    end
    
    if ~isfield(opts.fixed,'hprs')
        opts.fixed.hprs = 0;
    end
else
    opts.fixed.Z = 0;
    opts.fixed.prs = 0;
    opts.fixed.hprs = 0;
end

% check if verbose options
if ~isfield(opts,'verbose')
    opts.verbose = 1;
end

% check if maxiter options are supplied
if isfield(opts,'maxiter')
    if ~isfield(opts.maxiter,'EM')
        opts.maxiter.EM = 100;
    end
    
    if ~isfield(opts.maxiter,'Estep')
        opts.maxiter.Estep = 100;
    end
    
    if ~isfield(opts.maxiter,'MStep')
        opts.maxiter.MStep = 100;
    end
    
    if ~isfield(opts.maxiter,'hyperMstep')
        opts.maxiter.hyperMstep = 10;
    end
    
    if ~isfield(opts.maxiter,'inducingPointMstep')
        opts.maxiter.inducingPointMstep = 10;
    end

else
    opts.maxiter.EM = 100; % number of total EM iterations
    opts.maxiter.Estep = 100; % number of E step iterations if applicable
    opts.maxiter.Mstep = 100; % number of M step iterations if applicable
    opts.maxiter.hyperMstep = 20;  % number of hyper-Mstep iterations
    opts.maxiter.inducingPointMstep = 20; % number ofinduncing point-Mstep iterations 
end

% check if parallel option is supplied
if ~isfield(opts,'parallel')
    opts.parallel = 0;
end

% open parallel pool session with specified number of workers if applicable
if opts.parallel == 1
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    if isempty(poolobj) && ~isfield(opts,'numWorkers')
        warning('number of workers not specified, no paralell pool open -> running svGPFA sequentially')
        opts.numWorkers = 0;
    elseif isempty(poolobj) && isfield(opts,'numWorkers')
        parpool(opts.numWorkers); % open parpool with specified number of wokers
    else
        opts.numWorkers = poolobj.NumWorkers;
    end
end


% set rank of variational parameters for Poisson and Point Process
if ~strcmp(m.lik,'Gaussian')
    if ~isfield(opts,'varRnk')
       opts.varRnk = ones(1,m.dx);
    end
end

% set number of quadrature grid points to use
if strcmp(m.lik,'PointProcess')
    if ~isfield(opts,'nquad')
       opts.nquad = 50;
    end
end