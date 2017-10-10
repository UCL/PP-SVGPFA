function lik_poi = Lik_Poisson_svGPFA(m,mu_h,var_h,ntr);

if nargin < 4
    trEval = 1:m.ntr;
else
    trEval = ntr;
end

% account for zero padding
mask = permute(repmat(m.mask(:,trEval),1,1,m.dy),[1 3 2]);
mu_h(mask) = 0; % account for zero padding
var_h(mask) = 0;

intval = exp(mu_h + 0.5*var_h); % T x N x length(trEval)

lik_poi1 = m.BinWidth*sum(intval(:));
lik_poi2 = sum(vec(m.Y(:,:,trEval).*permute(mu_h,[2 1 3]))); 

lik_poi = -lik_poi1 + lik_poi2 - sum(log(factorial(vec(m.Y(:,:,trEval))))) + log(m.BinWidth);