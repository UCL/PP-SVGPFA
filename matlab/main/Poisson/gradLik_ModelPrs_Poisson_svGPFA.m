function grad = gradLik_ModelPrs_Poisson_svGPFA(m,C,b,mu_k,var_k);

mu_h = bsxfun(@plus,mtimesx(mu_k,C'), b');
var_h = mtimesx(var_k,(C.^2)');
mask = permute(repmat(m.mask,[1 1 m.dy]),[1 3 2]); % check this

intval = exp(mu_h + 0.5*var_h); % T x N x m.ntr
intval(mask) = 0;

grad_C = m.BinWidth*sum(mtimesx(intval,'T',mu_k),3) + m.BinWidth*(C.*sum(mtimesx(intval,'T',var_k),3));
grad_C2 = sum(mtimesx(m.Y,mu_k),3);
grad_b = m.BinWidth*sum(sum(intval,1),3)';
grad_b2 = sum(sum(m.Y,2),3);

grad_poi1 = [grad_C(:); grad_b];
grad_poi2 = [grad_C2(:); grad_b2];

grad = -grad_poi1 + grad_poi2;
