function m = svGPFA(m);
% m = svGPFA(m,maxiter);
%
% this function fits a sparse variational GPFA model to multivariate 
% observations 
%  
% model for observations:    y(t) ~  P(Y(t)| C1*f(t) + b1) 
%
% K latent processes:  f(t) = [f_1(t), ... , f_K(t)] 
% 
% input:
% ======
% m         -- model structure created using InitialiseModel_svGPFA
%
% output:
% ======
% m         -- model structure with optimised parameters
%
% See also: InitialiseModel_svGPFA
%
%
% Duncker, 2017
%
%
t_start = tic; % record starting time
abstol = 1e-05; % convergence tolerance
m.FreeEnergy = []; 

% print output
if m.opts.verbose
    fprintf('%3s\t%10s\t%10s\t%10s\n','iter','objective','increase','iterTime')
end

% ========= run variational inference =========
t_start_iter = tic;
for i = 1:m.opts.maxiter.EM
    
    % ========= precompute kernel matrices for E and M steps =====
    current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
    Kmats = m.EMfunctions.BuildKernelMatrices(m,current_hprs,m.Z,0); % get current Kernel matrices
    
    % ========= E-step: update variational parameters =========
    
    m = Estep(m,Kmats);
    
    % ========= compute new value of free energy ==================
    
    m.FreeEnergy(i,1) = VariationalFreeEnergy_svGPFA(m,Kmats);
    m.iterationTime(i,1) = toc(t_start_iter);
    
    if i > 1
        FEdiff = m.FreeEnergy(i,1) - m.FreeEnergy(i-1,1);
    else
        FEdiff = NaN;
    end
    
    if m.opts.verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n',i,m.FreeEnergy(i),FEdiff,m.iterationTime(i));
    end
    
    % ========= check convergence in free energy =========
    if i > 2 && abs(m.FreeEnergy(i,1) - m.FreeEnergy(i-1,1)) < abstol
        break;
    end
    t_start_iter = tic;
    % ========= M-step: optimise wrt model parameters =========
    
    if i > 1 % skip first M-step to avoid early convergence to bad optima
        m = Mstep(m,Kmats); 
    end
    
    % ========= hyper-M step: optimise wrt hyperparameters =========
    
    m = hyperMstep(m);

    % ========= inducing point hyper-M step: optimise wrt inducing point locations =========
    
    m = inducingPointMstep(m);

end

% save and report elapsed time
m.RunTime = toc(t_start);
if m.opts.verbose
    fprintf('Elapsed time is %1.5f seconds\n',m.RunTime);
end