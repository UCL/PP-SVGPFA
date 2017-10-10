function [cnts,dtnew] = discretiseSpikeTrain(sps,T,dt);
% function to convert spike trains into discretised Poisson counts in time
% bins of chosen width
%
% input:
% sps   {NxM}    --    spike trains for N neurons on M trials
% T              --    [0,T] is interval of experiment
% dt             --    bin width in units of time (same as T), should
%                      evenly divide T
%
% output:
% cnts  {NxM}   --     counts for N neurons on M trials
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nbins = T/dt;

if round(nbins) ~= nbins
    nbins = floor(T/dt);
    dtnew = T/nbins;
    fprintf('dt = %2.4f does not evenly divide T, adjusting width to dt = %1.4f \n',dt,dtnew)
else
    dtnew = dt;
end

cnts = cellfun(@(x)hist(x,linspace(0,T,nbins))',sps,'uni',0);