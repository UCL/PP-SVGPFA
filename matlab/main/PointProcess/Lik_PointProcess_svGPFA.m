function lik_pp = Lik_PointProcess_svGPFA(m,mu_h,var_h,ntr);

if nargin < 4
    trEval = 1:m.ntr;
    mu_hObsSum = mu_h(2,:);
    lik_pp2 = sum(cellvec(mu_hObsSum)); 
else
    trEval = ntr;
    mu_hObsSum = mu_h{2,1};
    lik_pp2 = sum(mu_hObsSum); 
end

mu_hQuad = mu_h{1,1};

intval = exp(mu_hQuad + 0.5*var_h); % numQuad x N x length(trEval)

lik_pp1 = sum(vec(mtimesx(m.wwQuad(:,:,trEval),'T',intval)));

lik_pp = -lik_pp1 + lik_pp2;
