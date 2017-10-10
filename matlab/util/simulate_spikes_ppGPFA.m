function [sps,rates] = simulate_spikes_ppGPFA(C,b,fs,T,nonlin,ngrid);
% simulate_spikes_ppGPFA -- function to simulate spike trains from a point
% process GPFA model
% input:
% -------
% C     (DxK) num       -- factor loadings
% b     (Kx1) num       -- offset
% fs    {KxM} cell      -- cell array containing function handles for 
%                          latent processes
% T     (1xM) vector    -- end point of interval [0,T] in which to generate
%                          spike train for each trial
% nonlin      handle    -- function handle for non-linearity 
% ngrid       scalar    -- number of points to use for evaluating intensity
%
% output:
% -------
% sps  {NxM} cell       -- spike train for each neuron, M trials
% rate {NxM} cell       -- array of function handles with rate functions
%                          for each neuron, M trials
% 
% Duncker, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[D,~] = size(C);
[~,ntr] = size(fs);
sps = cell(D,ntr);
rates = make_rates(C,b,fs,nonlin);

for m = 1:ntr
    for i = 1:D
        sps{i,m} = sample_inhomogenous(rates{i,m},T(m),ngrid)';
    end
end

end


function x = sample_inhomogenous(rr,T,ngrid)
% generate a nonhomogeneousl poisson process on [0,T] with intensity function intens
x = linspace(0,T,ngrid);
l = rr(x);
lam0 = max(l); % generate homogeneouos poisson process
u = rand(1,ceil(1.5*T*lam0));
x = cumsum(-(1/lam0)*log(u)); % points of homogeneous pp

x = x(x<T); n=length(x); % select those points less than T

l = rr(x); % evaluates intensity function

x = x(rand(1,n)<l/lam0); % filter out some points
end

function rates = make_rates(C,b,fs,nonlin)
% returns cell array with function handles to rates
[D,~] = size(C);
[~,ntr] = size(fs); % fs is function handle with latent for each trial
rates = cell(D,ntr);

for m = 1:ntr
    for d = 1:D
        rates{d,m} = @(tt) single_rate(tt,d,C,b,fs(:,m),nonlin);
    end
end

end

function rate_n  = single_rate(tt,n,C,b,fs,nonlin);
% returns rate for n-th neuron
[~,K] = size(C);
Ff = zeros(K,length(tt));
if ~isempty(Ff)
    for i = 1:K
        Ff(i,:)  = fs{i}(tt);
    end
end
rate_n = nonlin(C(n,:)*Ff + b(n));
end