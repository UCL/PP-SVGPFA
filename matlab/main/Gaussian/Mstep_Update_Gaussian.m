function m = Mstep_Update_Gaussian(m,Kmats);

% predict posterior means and variances
[mu_k, var_k] = predict_latentGPs_svGPFA(m,Kmats);

% mask to account for different trial lengths
mask = permute(repmat(m.mask,1,1,m.dx),[1 3 2]);
mu_k(mask) = 0; % account for zero padding
var_k(mask) = 0;

% update factor loadings
YMx = sum(sum(mtimesx(permute(bsxfun(@minus,m.Y,m.prs.b),[1 4 2 3]),permute(mu_k,[4 2 1 3])),4),3);
MMx = diag(sum(sum(var_k,1),3)) + sum(mtimesx(permute(mu_k,[2 1 3]),mu_k),3);
C = YMx / MMx;

% update constant offset
b =  sum(reshape((m.Y - mtimesx(m.prs.C,permute(mu_k,[2 1 3])))./permute(m.trLen,[3 1 2]),m.dy,[]),2)./m.ntr;

% get posterior expectations of linear predictor
Oii =  mtimesx(m.prs.C.^2,permute(var_k,[2 1 3]));
Nui =  mtimesx(m.prs.C,permute(mu_k,[2 1 3]));
Nui =  bsxfun(@plus,Nui,m.prs.b);

mask =  bsxfun(@(x,y) x < y,repmat(m.trLen,size(m.Y,2),1), (1:size(m.Y,2))');
mask = permute(repmat(mask,1,1,m.dy),[3 1 2]);
Nui(mask) = 0;

% update noise variances
psi = ((m.Y -  Nui).^2 + Oii)./permute(m.trLen,[3 1 2]);
psi = sum(psi(:,:),2)./m.ntr;

% update model structure
m.prs.C = C;
m.prs.b = b;
m.prs.psi = psi;