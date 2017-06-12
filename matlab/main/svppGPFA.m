function m = svppGPFA(T,Y,kerns,Z,C,b,nonlin,ng,rnk,maxiter,varargin);
% m = svppGPFA(T,sps,kerns,Z,C,b,nonlin,ng);
% this function fits a sparse variational point process GPFA model to spike
% time observations of multiple neurons.
%  
% model for firing rate: lambda(t) = g( C*f(t) + b ) 
%
% K latent processes:  f(t) = [f_1(t), ... , f_K(t)] 
%
% and lambda(t) = [lambda_1(t), ... , lambda_N(t)] vector over neurons
%
% INPUT:
% =========================================================================
%
%   T       -- [0,T] is the time interval in the experiment
%
%   sps     -- {N,M} cell array containing C(n)x1 dimentional spike trains
%               where C(n) is the total number of spikes in [0,T] for
%               neuron n (N = neurons, M = trials)
%
%   kerns   -- {K,1} cell array containing kernel function handles created 
%              by calling 
%              kern1 =  buildKernel(name,hprs); 
%              kerns = {kern1, ...}
%
%   Z       -- {K,M} cell array containing the inducing points for each
%              latent process, Z = {Z1,Z2,...}. Zi are Nz_i x 1 vectors
%
%   C       -- (NxK) matrix with initial values of factor loadings
%
%   b       -- (Nx1) vector with initial values of constant offset
%
%   nonlin  -- object specifying nonlinearity. Created by calling
%              nonlin =  buildNonlin('Exponential'); (others to follow in
%              later implementations)
%
%   ng      -- scalar value for the number of grid points to use for
%              quadrature
%
%   rnk     -- rank of variational covarainces parameterised as low-rank +
%              diagonal matrix. vector for each latents e.g [1 1 2]
%                 i.e. S_k = LL' + diag(l)
%                      number of columns in L is specified by entries in rnk 
%
%   maxiter -- maximum number of iterations of variational EM routine
%
%   fixed   -- optional structure containing variables that should remain 
%              fixed during optimization, possible fields:
%                   fixed.Z (for inducing points)
%                   fixed.prs (for C and b) 
%                   fixed.hprs (for kernel hyperparameters)
%              currently it is only possible to fix all or none for group of
%              parameters
%
% OUTPUT:
% =========================================================================
% fit      -- structure containing model fit and convergence diagnostics
%
% Duncker, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_start = tic;
abstol = 1e-05; % convergence tolerance

% ========= initialise model structure =========

m.T = T;
m.Y = Y;
m.kerns = kerns;
m.Z = Z;
m.C = C;
m.b = b;
m.ng = ng;
m.dx = size(C,2);
m.dy = size(C,1);
m.ntr = size(Y,2);
m.nonlin = nonlin;
m.num_inducing = cellfun(@(x) max(size(x)),Z(:,1))';
m.rnk = rnk;
% check if anything should be fixed and corresponding update skipped

if nargin > 10 && isfield(varargin{1},'Z')
    m.fixed.Z = 1;
else
    m.fixed.Z = 0;
end
if nargin > 10 && isfield(varargin{1},'prs')
    m.fixed.prs = 1;
else
    m.fixed.prs = 0;
end

if nargin > 10 && isfield(varargin{1},'hprs')
    m.fixed.hprs = 1;
else
    m.fixed.hprs = 0;
end


% ========= initialise variational parameters =========

for ii = 1:m.dx
    m.q_mu{ii} = zeros(m.num_inducing(ii),1,m.ntr);
    m.q_sqrt{ii} = repmat(vec(0.01*eye(m.num_inducing(ii),rnk(ii))),[1 1 m.ntr]);
    m.q_diag{ii} = repmat(0.01*ones(m.num_inducing(ii),1),[1 1 m.ntr]);
end

m.tt = linspace(0,m.T,m.ng)'; % grid of time points
m.epsilon = 1e-05; % jitter level for numerical stability
m.FreeEnergy = [];

for numtr = 1:m.ntr % indices to be used when concatenating all spikes for each trial
    m.neuronIndex{numtr} = cellvec(cellfun(@(x,y)repmat(x,[y 1]),num2cell(1:m.dy),(cellfun(@length,m.Y(:,numtr),'uni',0))','uni',0));
end

% print output
fprintf('%3s\t%10s\t\n', 'iter', 'objective');

% ========= run variational inference =========

for i = 1:maxiter
    % ========= E-step: optimise wrt variational parameters =========
    
    m = Estep(m); % update q_mu and q_sqrt for all trials
    
    % report current value of free energy
    fprintf('%3d\t%10.4f\n', i, m.FreeEnergy(i));
    
    % check convergence in free energy
    if i > 2 && abs(m.FreeEnergy(i) - m.FreeEnergy(i-1)) < abstol
        break;
    end

    % ========= M-step: optimise wrt model parameters =========
    
    if i > 1 % skip first M-step to avoid early convergence to bad optima
        m = Mstep(m); % update b and C
    end
    
    % ========= hyper-M step: optimise wrt hyperparameters =========
    
    m = hyperMstep(m);
    
    % ========= inducing point hyper-M step: optimise wrt inducing point locations =========
    
    m = inducingPointMstep(m);
    
end

% report elapsed time
m.RunTime = toc(t_start);
