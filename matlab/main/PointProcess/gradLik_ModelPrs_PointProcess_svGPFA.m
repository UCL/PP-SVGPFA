function grad = gradLik_ModelPrs_PointProcess_svGPFA(m,C,b,mu_k,var_k);

mu_hQuad = bsxfun(@plus,mtimesx(mu_k{1},C'), b');

var_h = mtimesx(var_k,(C.^2)');

intval = exp(mu_hQuad + 0.5*var_h); % T x N x m.ntr

quadIntval = bsxfun(@times,m.wwQuad,intval);
gradCtt1 = mtimesx(quadIntval,'T',mu_k{1});
gradCtt2 = permute(mtimesx(permute(mtimesx(permute(var_k,[1 4 2 3]),...
    permute(C,[3 1 2 4])),[5 1 2 3 4]),permute(quadIntval,[1 5 2 4 3])),[3 4 5 1 2]);

grad_C = gradCtt1 + gradCtt2;

grad_b = permute(mtimesx(m.wwQuad,'T',intval),[2 3 1]);
for nn =  1:m.ntr
    mm1 = m.spikecellID{nn};
    grad_C2(:,:,nn) = mm1'*mu_k{2,nn};
    grad_b2(:,nn) = sum(mm1,1)';
end

grad_poi1 = sum([reshape(grad_C,[],m.ntr); grad_b],2);
grad_poi2 = sum([reshape(grad_C2,[],m.ntr); grad_b2],2);

grad = -grad_poi1 + full(grad_poi2);


