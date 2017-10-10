function KMats = BuildKernelMatrices_Gaussian_svGPFA(m,hprs,Z,flag,ntr);
% build the kernel matrices needed for Gaussian svGPFA
% flag indicates whether gradient matrices should also be returned
% flag = 0 - no gradients
% flag = 1 - gradients wrt hyperparameters
% flag = 2 - gradeints wrt inducing points
%
if nargin < 5 % evalutate over all trials
    trEval = 1:m.ntr;
    timeEval = 1:m.ntr;
else
    trEval = 1;
    timeEval = ntr;
end

% pre-allocate cell array for saving kernel matrices
Kzz = cell(m.dx,1);
Kzzi = cell(m.dx,1);
Ktz = cell(m.dx,1);
Ktt = zeros(length(m.tt),m.dx);

% pre-allocate space for gradients
if flag == 1
    dKtzhprs = cell(m.dx,1);
    dKtt = cell(m.dx,1);
    dKzzhprs = cell(m.dx,1);
elseif flag ==2
    dKzzin = cell(m.dx,1);
    dKtzin = cell(m.dx,1);
end

% for k = 1:m.dx
%     if flag == 1
%         [Ktt(:,k),dKtt{k}] = m.kerns{k}.Kdiag(hprs{k},m.tt);
%     else
%         Ktt(:,k) = m.kerns{k}.Kdiag(hprs{k},m.tt);
%     end
%     
%     for numtr = trEval
%         Kzz{k}(:,:,numtr) = m.kerns{k}.K(hprs{k},Z{k}(:,:,numtr)) + m.epsilon*eye(m.numZ(k));
%         Kzzi{k}(:,:,numtr) = pinv(Kzz{k}(:,:,numtr));
%         Ktz{k}(:,:,numtr) = m.kerns{k}.K(hprs{k},m.tt,Z{k}(:,:,numtr));
%         if flag == 1
%             dKzzhprs{k}(:,:,:,numtr) = m.kerns{k}.dKhprs(hprs{k},Z{k}(:,:,numtr));
%             dKtzhprs{k}(:,:,:,numtr) = m.kerns{k}.dKhprs(hprs{k},m.tt,Z{k}(:,:,numtr));
%         elseif flag ==2
%             dKtzin{k}(:,:,:,numtr) = m.kerns{k}.dKin(m.kerns{k}.hprs,m.tt,Z{k}(:,:,numtr));
%             [~,dKzzin{k}(:,:,:,numtr)] = m.kerns{k}.dKin(m.kerns{k}.hprs,Z{k}(:,:,numtr));
%         end
%     end
% end

for k = 1:m.dx
    if flag == 1
        [Ktt(:,k),dKtt{k}] = m.kerns{k}.Kdiag(hprs{k},m.tt);
    else
        Ktt(:,k) = m.kerns{k}.Kdiag(hprs{k},m.tt);
    end
    
    Kzz{k} = m.kerns{k}.K(hprs{k},Z{k}(:,:,trEval)) + m.epsilon*eye(m.numZ(k),m.numZ(k));
    for numtr = trEval
        Kzzi{k}(:,:,numtr) = pinv(Kzz{k}(:,:,numtr));
    end
    Ktz{k} = m.kerns{k}.K(hprs{k},m.tt,Z{k});
    if flag == 1
        dKzzhprs{k} = m.kerns{k}.dKhprs(hprs{k},Z{k});
        dKtzhprs{k} = m.kerns{k}.dKhprs(hprs{k},m.tt,Z{k});
    elseif flag ==2
        dKtzin{k} = m.kerns{k}.dKin(m.kerns{k}.hprs,m.tt,Z{k});
        [~,dKzzin{k}] = m.kerns{k}.dKin(m.kerns{k}.hprs,Z{k});
    end
end

KMats.Kzz = Kzz;
KMats.Kzzi = Kzzi;
KMats.Ktz = Ktz;
KMats.Ktt = Ktt;

if flag == 1
    KMats.dKzzhprs = dKzzhprs;
    KMats.dKtzhprs = dKtzhprs;
    KMats.dKtt = dKtt;
elseif flag == 2
    KMats.dKtzin = dKtzin;
    KMats.dKzzin = dKzzin;
end
