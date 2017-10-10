%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
rng(2)
%% make latents
dx = 3; % number of latents 
dy = 50; % number of neurons
ntr = 14; % number of trials; ntr > 1 activates parpool session in matlab
T = repmat(20,ntr,1); % maximum time for each trial
% T = [19 20];

trLen = T;
% T = [18 17 20 20 19];
fq1 = 2.5; % frequency of oscillation
fq2 = 2.5;

% build latent functions
fs = cell(dx,ntr);
for ii = 1:ntr 
    fs{1,ii}  = @(t)  0.8*exp(-2*(sin(pi*abs(t)*fq1).^2)/5).*sin(2*pi*fq1*t);
    fs{2,ii}  = @(t)  0.5*cos(2*pi*fq2*t);
    fs{3,ii}  = @(t)  0.7*exp(-0.5*(t-5).^2/8).*cos(2*pi*.12*t)  + 0.8*exp(-0.5*(t-10).^2/12).*sin(2*pi*t*0.1 + 1.5);
end

%% simulate spike train
% model parameters
prs.C = bsxfun(@times,randn(dy,dx),[1 1.2 1.3]);
prs.b = 0.2*ones(dy,1);
% nonlinearity
nonlin = @(x) exp(x);
% grids for plotting
ngrid = 2000; 
t = linspace(0,max(T),ngrid); % time grid for sampling spike train
% generate spikes
[sps,rates] = simulate_spikes_ppGPFA(prs.C,prs.b,fs,T,nonlin,ngrid);
Y = cell(ntr,1);
for nn = 1:ntr; Y{nn} = sps(:,nn);end; % this is just to keep things consitent with other inputs
% print total number of spikes across neurons and plot rates and raster
fprintf('total number of spikes across all neurons: %d \n',sum(cellfun(@(x)size(x,1),sps)))
figure;
subplot(121);plotRaster(sps); title('raster plot of spike times')
subplot(122);hold on; for i = 1:dy; plot(rates{i,1}(t));end
title('example firing rates')
%% set up fitting structure
nZ(1) = 10; % number of inducing points for each latent
nZ(2) = 11;
nZ(3) = 12;
Nmax = 500;
dt = max(T)/Nmax;

% set up kernels
kern1 = buildKernel_svGPFA('Periodic',[0.5;1.5;1/fq1]);
kern2 = buildKernel_svGPFA('Periodic',[0.5;1.2;1/fq2]);
kern3 = buildKernel_svGPFA('RBF',[0.5;1]);
kerns = {kern1, kern2,kern3};

% set up list of inducing point locations
Z = cell(dx,1);
for ii = 1:dx
    for jj = 1:ntr
        Z{ii}(:,:,jj) = linspace(dt,T(jj),nZ(ii))';
    end
end

%% initialise model structure
options.parallel = 0;
options.verbose = 1;
options.nquad = 300;
m = InitialiseModel_svGPFA('PointProcess',Y,trLen,kerns,Z,prs,options);

%% set extra options and fit model
m.opts.maxiter.EM = 50; % maximum number of iterations to run
m.opts.fixed.Z = 1;
m = svGPFA(m);

%% predict latents and MultiOutput GP
ngtest = 2000;
testTimes = linspace(0,max(T),ngtest)';
pred = predictNew_svGPFA(m,testTimes);
%% plot latents for a given trial
nn = 2;
figure; 
for ii = 1:dx
    subplot(3,1,ii);plot(testTimes,fs{ii,nn}(testTimes),'k','Linewidth',1.5);
    hold on; plot(testTimes,pred.latents.mean(:,ii,nn),'Linewidth',1.5);
    errorbarFill(testTimes, pred.latents.mean(:,ii,nn), sqrt(pred.latents.variance(:,ii,nn)));
    hold on; plot(m.Z{ii}(:,:,nn),min(pred.latents.mean(:,ii,nn))*ones(size(m.Z{ii}(:,:,nn))),'r.','markersize',12)
    box off;
    xlim([0 max(T)])
    if ii ~= dx
       set(gca,'Xtick',[]) 
    end
    set(gca,'TickDir','out')
end
xlabel('time')

%% plot all estimated vs true log-rates / Gaussian means
latents_nn = cell2mat(cellfun(@(x)feval(x,testTimes),fs(:,nn),'uni',0)')';
figure;
hold on;plot(pred.multiOutputGP.mean(:,:,nn)',prs.C*latents_nn + prs.b,':','Linewidth',1.8);
hold on;
plot(linspace(-5,5,100),linspace(-5,5,100),'k')
xlabel('estimated mean')
ylabel('true mean')
title(sprintf('corr = %1.3f',corr(vec(pred.multiOutputGP.mean(:,:,nn)'),vec(prs.C*latents_nn + prs.b))))