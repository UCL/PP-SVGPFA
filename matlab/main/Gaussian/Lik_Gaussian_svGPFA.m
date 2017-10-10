function lik_gg = Lik_Gaussian_svGPFA(m,mu_h,var_h,ntr);

if nargin < 4
    trEval = 1:m.ntr;
    saveIdx = 1:m.ntr;
    timeEval = 1:max(m.trLen);
    R = sum(m.trLen);
else
    trEval = ntr;
    saveIdx = 1;
    timeEval = 1:m.trLen(ntr);
    R = m.trLen(ntr);
end

t1_gg = -0.5*R*sum(log(m.prs.psi)) - 0.5*(R*m.dy)*log(2*pi) - 0.5*sum(vec(mtimesx(m.Y(:,timeEval,trEval).^2,'T',1./m.prs.psi))); % constants wrt variational prs

t2_gg = -0.5*sum(vec(mtimesx(var_h(timeEval,:,saveIdx),1./m.prs.psi))); % terms depending on variational covariance

t3_gg = sum(vec(mtimesx(mu_h(timeEval,:,saveIdx).*permute(m.Y(:,timeEval,trEval),[2 1 3]) - 0.5*mu_h(timeEval,:,saveIdx).^2,1./m.prs.psi))); % terms depending on variational mean

lik_gg =  t1_gg + t2_gg + t3_gg;
