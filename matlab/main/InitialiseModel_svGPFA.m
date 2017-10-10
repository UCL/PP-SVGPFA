function m = InitialiseModel_svGPFA(lik,Y,trLen,kerns,Z,prs,options);
% m = InitialiseModel_svGPFA(lik,Y,trLen,kerns,Z,prs,options);
% 
% function to initialise the model structure for sparse variational GPFA
% 
% input:
% =======
% lik         -- string taking values {'Gaussian','Poisson','PointProcess'}
%                specifying the type of observations of the svGPFA model
% Y     {M}   -- a cell array containing, for the mth trial
%                   - Gaussian:     [D x N(m)] array where N(m) are the 
%                                   number of samples in the mth trial
%                   - Poisson:      [D x N(m)] array where N(m) is
%                                   trLen(m)/binWidth
%                   - PointProcess: {D} cell array containing the
%                                   event-time observations for each of D 
%                                   point processes
%
% trLen (Mx1) -- vector specifying the length (in units of time) of each of
%                the M trials
%
% kerns {1,K}   -- a cell array containing the kernel function hanfles for 
%                each of K latent processes. This should be created as
%                   kern{k} =  buildKernel_svGPFA(name,hprs); 
%                   
% Z    {K,M}  -- cell array containing the inducing points for each of K
%                latent process, Z{k,m} is a  Nz(k) x 1 vectors
%
% prs         -- structure containing the initial parameter values
%                   - prs.C      factor loadings
%                   - prs.b      constant offset
%                   - prs.psi    uniquenesses (for Gaussian )
%
% options     -- structure with additional optional values to be declared
%                 .parallel = integer in {0,1} to indicate whether to run
%                             updates in parallel across trials using 
%                             MATLAB's Parallel Processing Toolbox 
%                 .nquad    = number of points for quadrature (PointProcess)
%                 .fixed    = .Z        set to 1 to fix sets of variables
%                             .prs
%                             .hprs
%                 
%
% output:
% =======
% m         -- model structure to be passed into svGFA
%
%
% See also: svGPFA, buildKernel_svGPFA
%
%
% Duncker, 2017
%
%

% check that kerns are M x 1
if size(kerns,1) > 1
    kerns = kerns';
end

m.lik = lik;
m.Z = Z;
m.kerns = kerns;
m.prs = prs;
m.ntr = length(trLen); % number of trials
m.epsilon = 1e-05; % value of diagonal added to kernel inversion for stability
m.dx = size(prs.C,2);
m.dy = size(prs.C,1); % number of pixels in calcium trace
m.numZ = cellfun(@(x) size(x,1),Z)'; % number of inducing points

% set default option values or supplied ones
opts = setOptionValues_svGPFA(m,options);
m.opts = opts;

% construct likelihood dependent function handles
m = setFunctionHandles_svGPFA(m,lik);

% set other likelihood dependent things
switch lik
    case 'Gaussian'
        % arrange data into appropriate format
        [Ystack,trLenN,mask,tsamp,BinWidth] = arrangeData_Gaussian_svGPFA(m,Y,trLen);
        m.Y = Ystack; % zero padded tensor
        m.mask = mask; % mask indicating where trials end
        m.trLen = trLenN; % trial length in terms of size of data, not time
        m.tt = tsamp'; % time points at which data is sampled [0, max(trLen)]
        m.BinWidth = BinWidth;
        
    case 'Poisson'
        % arrange data into appropriate format
        [Ystack,trLenN,mask,tsamp,BinWidth] = arrangeData_Poisson_svGPFA(m,Y,trLen);
        m.Y = Ystack; % zero padded tensor
        m.mask = mask; % mask indicating where trials end
        m.trLen = trLenN; % trial length in terms of size of data, not time
        m.tt = tsamp'; % time points at which data is sampled [0, max(trLen)]          
        m.BinWidth = BinWidth;
        m = initialise_VariationalPrs_svGPFA(m);
        
    case 'PointProcess'
        % arrange data into appropriate format
        [Ystack,index] = arrangeData_PointProcess_svGPFA(m,Y);
        m.Y = Ystack; % stacked over all output dimensions
        m.index = index; % index for remembering which observation belongs to which output
        m.trLen = trLen; % trial lengths in units of time
        for nn = 1:m.ntr
            [ttQuad(:,:,nn),wwQuad(:,:,nn)] = legquad(opts.nquad,0,m.trLen(nn)); % calculate weights and nodes for Gauss-Legendre quadrature
             spikecellID{nn} = bsxfun(@(x,y) x == y, m.index{nn}, sparse(1:m.dy));

        end
        % rescale quadrature weights to be in correct interval for each trial:
        m.ttQuad = permute(ttQuad,[2 1 3]); % nquad x 1 x M
        m.wwQuad = permute(wwQuad,[2 1 3]); % nquad x 1 x M
        m.spikecellID = spikecellID;
        m = initialise_VariationalPrs_svGPFA(m);
        
end

