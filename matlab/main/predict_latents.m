function [mu_k,var_k] = predict_latents(m,Tnew,ntr,varargin)
% predictive distribution over weighted sum of GPs
K = m.dx;

mu_k= zeros(length(Tnew),K);
var_k = zeros(length(Tnew),K);

if nargin > 3
    Kzz = varargin{1};
    Ktz = varargin{2};
    Ktt = varargin{3};
end

for k = 1:K
    if nargin > 3
        [mu_k(:,k), var_k(:,k)] = predict_single(m,Tnew,k,ntr,Kzz{k,ntr},Ktz{k,ntr},Ktt{k,ntr});
    else
        [mu_k(:,k), var_k(:,k)] = predict_single(m,Tnew,k,ntr);
    end
end

end