%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
rng(2)
%% make latents
dx = 3; % number of latents 
dy = 100; % number of neurons
T = 20; % maximum time
ntr = 1; % number of trials; ntr > 1 activates parpool session in matlab

fq1 = 2.5; % frequency of oscillation
fq2 = 2.5;

% build latent functions
fs = cell(dx,ntr);
for ii = 1:ntr 
    fs{1,ii}  = @(t)  0.8*exp(-2*(sin(pi*abs(t)*fq1).^2)/5).*sin(2*pi*fq1*t);
    fs{2,ii}  = @(t)  0.5*cos(2*pi*fq2*t);
    fs{3,ii}  = @(x)  0.7*exp(-0.5*(x-5).^2/8).*cos(2*pi*.12*x)  + 0.8*exp(-0.5*(x-10).^2/12).*sin(2*pi*x*0.1 + 1.5);
end

%% simulate spike train
% model parameters
C = bsxfun(@times,randn(dy,dx),[1 1.2 1.3]);
b = 0.2*ones(dy,1);
% nonlinearity
nonlin = @(x) exp(x);
% grids for plotting
ngrid = 2000; 
t = linspace(0,T,ngrid); % time grid for sampling spike train
% generate spikes
[sps,rates] = simulate_spikes_ppGPFA(C,b,fs,T,nonlin,ngrid);

% print total number of spikes across neurons and plot rates and raster
fprintf('total number of spikes across all neurons: %d \n',sum(cellfun(@(x)size(x,1),sps)))
figure;
subplot(121);plotRaster(sps); title('raster plot of spike times')
subplot(122);hold on; for i = 1:dy; plot(rates{i,1}(t));end
title('example firing rates')
%% set up fitting structure
nZ1 = 15; % number of inducing points for each latent
nZ2 = 15;
nZ3 = 10;
ng = 500; % number of grid points for numerical integration
 
% set up kernels
kern1 = buildKernel('Periodic',[0.5;1.5;1/fq1]);
kern2 = buildKernel('Periodic',[0.5;1.2;1/fq2]);
kern3 = buildKernel('RBF',[0.5;1]);
kerns = {kern1, kern2,kern3};

% set up non-linearity
nonlin = buildNonlin('Exponential');

% set up list of inducing point locations
Z = cell(dx,ntr);
Z(1,:) = {linspace(.2,T,nZ1)'};
Z(2,:) = {linspace(.2,T,nZ2)'};
Z(3,:) = {linspace(.2,T,nZ3)'};

%% run ppGPFA optimization for all trials
maxiter = 50; % maximum number of iterations to run
rnk = [1 1 1]; % rank of variational covariance parameterization S = L*L' + diag

% to fix groups of parameters to their initial value:
% clear fixed;
% fixed.Z = 1; % fix inducing point locations
% fixed.prs = 1; % fix model parameters
% fixed.hprs = 1; % fix kernel hyperparameters
% m = svppGPFA(T,sps,kerns,Z,C,b,nonlin,ng,rnk,maxiter,fixed);

% to optimise over everything:
m = svppGPFA(T,sps,kerns,Z,C,b,nonlin,ng,rnk,maxiter);

%% predict latents and firing rates 
mu_fs=cell(ntr,1);
var_fs = cell(ntr,1);
mu_h=cell(ntr,1);
var_h = cell(ntr,1);

for nn = 1:ntr
    [mu_fs{nn},var_fs{nn}] = predict_latents(m,t',nn);
    [mu_h{nn}, var_h{nn}] = map_to_nontransformed_rate(m,mu_fs{nn},var_fs{nn});
end

%% plot latents for a given trial
nn = 1;
figure; 
for ii = 1:dx
    subplot(3,1,ii);plot(linspace(0.0, 20.0, 1000),fs{ii,nn}(linspace(0.0, 20.0, 1000)),'k','Linewidth',1.5);
    hold on; plot(linspace(0.0, 20.0, 1000),mu_fs{nn}(:,ii),'Linewidth',1.5);
    errorbarFill(linspace(0.0, 20.0, 1000), mu_fs{nn}(:,ii), sqrt(var_fs{nn}(:,ii)));
    hold on; plot(m.Z{ii,nn},min(mu_fs{nn}(:,ii))*ones(size(m.Z{ii,nn})),'r.','markersize',12)
    box off;
    xlim([0 T])
    if ii ~= dx
       set(gca,'Xtick',[]) 
    end
    set(gca,'TickDir','out')
end
xlabel('time')
%% plot rates for a given trial and neuron
nn = 1;
figure; plot(linspace(0.0, 20, 1000),exp(mu_h{nn}(:,1:10)))
i = 12;
plot(linspace(0.0, 20, 1000),exp(mu_h{nn}(:,i))); hold on;plot(linspace(0.0, 20, 1000),rates{i}(t),'k--');
errorbarFill(linspace(0.0, 20, 1000), exp(mu_h{nn}(:,i)), ...
    sqrt((exp(var_h{nn}(:,i))-1).*(exp(2*mu_h{nn}(:,i) + var_h{nn}(:,i)))));
ylabel('spikes/sec')
xlabel('time in seconds')
title(sprintf('neuron number %d',i))
hold off;
%% plot all estimated vs true log-rates
figure;hold on;for i = 1:20; plot((mu_h{nn}(:,i)),log(rates{i}(t)),':','Linewidth',1.8);end
hold on;
plot(linspace(-5,5,100),linspace(-5,5,100),'k')
xlabel('estimated log-rate')
ylabel('true log-rate')