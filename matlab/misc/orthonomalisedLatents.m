function [mu_fs_orth,Corth] = orthonomalisedLatents(C,mu_fs,ntr)
% function to predict latent function given model fit and orthonormalize
% basis for consistency. 
% input:
% m  -  is a structure contianing the model
% t  -  is a row vector of time points where GP should be evaluated
% output:
% mu_fs_orth
%
%
[U,S,V] = svd(C);
dx = size(C,2);
mu_fs_orth=cell(ntr,1);
% var_fs_Orth = cell(ntr,1);
if ~iscell(mu_fs)
    mu_fs = {mu_fs};
end
for nn = 1:ntr
    mu_fs_orth{nn} = mu_fs{nn}*V*sqrt(S(1:dx,:)');
%     for i = 1:dx
%         var_fs_Orth{nn}(:,i) = diag(sqrt(S)*V'*diag(var_fs{nn}(:,i))*V*sqrt(S'));
%     end
end

Corth = U;
