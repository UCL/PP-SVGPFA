function [mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats,ntr);

if nargin < 3
    trEval = 1:m.ntr;
    saveIdx = 1:m.ntr;
else
    trEval = ntr;
    saveIdx = 1;
end

% Npred = size(Kmats.Ktz{1},1); % number of points where mean and covariance are evaluated
% mu_k = zeros(Npred,m.dx,length(trEval));
% var_k = zeros(Npred,m.dx,length(trEval));

for k = 1:m.dx
    
    q_mu_k = m.q_mu{k}(:,:,trEval); % now 3D over trials
    q_sigma_k = m.q_sigma{k}(:,:,trEval);
        
    Ak = mtimesx(Kmats.Kzzi{k}(:,:,saveIdx),q_mu_k);
    
    % check if we have two different kinds of evaluations, due to
    % Quadrature:
    
    if isfield(Kmats,'Quad') % need mean and variance for quad, mean for observed spikes
        Bkf = mtimesx(Kmats.Kzzi{k}(:,:,saveIdx),Kmats.Quad.Ktz{k}(:,:,saveIdx),'T');
        mm1f = mtimesx(q_sigma_k - Kmats.Kzz{k}(:,:,saveIdx),Bkf);
        mu_k{1,1}(:,k,saveIdx) = mtimesx(Kmats.Quad.Ktz{k}(:,:,saveIdx),Ak);
        for nn = 1:length(trEval)
            mu_k{2,nn}(:,k) = Kmats.Obs.Ktz{k,saveIdx(nn)}*Ak(:,:,saveIdx(nn));
        end
        var_k(:,k,saveIdx) = bsxfun(@plus,Kmats.Quad.Ktt(:,k), sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));
    else % if no quadrature just evalute on a grid
        Bkf = mtimesx(Kmats.Kzzi{k}(:,:,saveIdx),Kmats.Ktz{k}(:,:,saveIdx),'T');
        mm1f = mtimesx(q_sigma_k - Kmats.Kzz{k}(:,:,saveIdx),Bkf);
        mu_k(:,k,saveIdx) = mtimesx(Kmats.Ktz{k}(:,:,saveIdx),Ak);
        var_k(:,k,saveIdx) = bsxfun(@plus,Kmats.Ktt(:,k), sum(bsxfun(@times,permute(Bkf,[2 1 3]),permute(mm1f,[2 1 3])),2));
    end
end
