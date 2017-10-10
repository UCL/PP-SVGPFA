function m = Estep_Update_Gaussian(m,Kmats);
% function to update the variational mean and variance for a single trial
Kzzi = Kmats.Kzzi;
Ktz = Kmats.Ktz;
q_sigma_inv = cell(m.dx,1);
q_sigma = cell(m.dx,1);

% loop over latent processes and update each variational covariance
for k = 1:m.dx
    Ak = mtimesx(Kzzi{k},Ktz{k},'T');
    mask = repmat(permute(m.mask,[3 1 2]),[m.numZ(k) 1 1]);
    Ak(mask) = 0;
    KzziKztOut = mtimesx(Ak,Ak,'T');  % Kzzi*Kzt*Ktz*Kzzi
    Km{k} = permute(Ak,[4 1 2 3]);
    % variational precision
    q_sigma_inv{k} = (Kzzi{k} + sum(m.prs.C(:,k).^2./m.prs.psi)*KzziKztOut);
end

% update all variational means simultaneously
KzziBlk = blkdiagND(Kzzi{:});
KmBlk = blkdiagND(Km{:});
CtPsii = bsxfun(@times, m.prs.C',1./m.prs.psi'); % C'*inv(psi)

KCtPiC = mtimesx(KmBlk,'T',CtPsii*m.prs.C);
KKcp = mtimesx(KCtPiC,KmBlk);

for nn = 1:m.ntr % need to solve ntr systems of equations
    
    Mx1 = (sum(KKcp(:,:,1:m.trLen(nn),nn),3) + KzziBlk(:,:,nn));
    Mx2 = permute(sum(mtimesx(mtimesx(KmBlk(:,:,1:m.trLen(nn),nn),'T',CtPsii),permute((m.Y(:,1:m.trLen(nn),nn) - m.prs.b),[1 3 2])),3),[1 2 4 3]);

    q_mu_nn = Mx1\Mx2;
    
    for k = 1:m.dx
        q_sigma{k}(:,:,nn) = inv(q_sigma_inv{k}(:,:,nn));
    end
    % reshape q_mu_all into cell for each q_mu
    q_mu_kn(:,nn) = mat2cell(q_mu_nn,m.numZ,1);
end

for k = 1:m.dx
    q_mu{k} = permute(cell2mat(q_mu_kn(k,:)),[1 3 2]);
end
    

m.q_mu = q_mu;
m.q_sigma = q_sigma;