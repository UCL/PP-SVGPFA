function KMats = BuildKernelMatrices_PointProcess_svGPFA(m,hprs,Z,flag,ntr);
% build the kernel matrices needed for Gaussian svGPFA
% flag indicates whether gradient matrices should also be returned
% flag = 0 - no gradients
% flag = 1 - gradients wrt hyperparameters
% flag = 2 - gradients wrt inducing points
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
KtzQuad = cell(m.dx,1);
KttQuad = zeros(size(m.ttQuad,1),m.dx,length(trEval));
KtzObs = cell(m.dx,length(trEval));

% pre-allocate space for gradients
if flag == 1
    dKtzQuadhprs = cell(m.dx,1);
    dKttQuad = cell(m.dx,1);
    dKzzhprs = cell(m.dx,1);
    
    dKtzObshprs = cell(m.dx,length(trEval));
    
elseif flag == 2
    dKzzin = cell(m.dx,1);
    dKtzQuadin = cell(m.dx,1);
    dKtzObsin = cell(m.dx,length(trEval));
end


for k = 1:m.dx
    if flag == 1
        [KttQuad(:,k,:),dKttQuad{k}] = m.kerns{k}.Kdiag(hprs{k},m.ttQuad(:,:,timeEval));
    else
        KttQuad(:,k,:) = m.kerns{k}.Kdiag(hprs{k},m.ttQuad(:,:,timeEval));
    end
    
    Kzz{k} = m.kerns{k}.K(hprs{k},Z{k}(:,:,trEval)) + m.epsilon*eye(m.numZ(k),m.numZ(k));
    
    for nn = 1:length(trEval)
        Kzzi{k}(:,:,trEval(nn)) = pinv(Kzz{k}(:,:,trEval(nn)));
        KtzObs{k,nn} = m.kerns{k}.K(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,trEval(nn)));
    end
    
    
    KtzQuad{k} = m.kerns{k}.K(hprs{k},m.ttQuad(:,:,timeEval),Z{k});
    
    if flag == 1
        dKzzhprs{k} = m.kerns{k}.dKhprs(hprs{k},Z{k});
        
        for nn = 1:length(trEval)
            dKtzObshprs{k,nn} =  m.kerns{k}.dKhprs(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,trEval(nn)));
        end
        
        dKtzQuadhprs{k} = m.kerns{k}.dKhprs(hprs{k},m.ttQuad(:,:,timeEval),Z{k});
    elseif flag ==2
        for nn = 1:length(trEval)
            dKtzObsin{k,nn} = m.kerns{k}.dKin(hprs{k},m.Y{timeEval(nn)},Z{k}(:,:,trEval(nn)));
        end
        dKtzQuadin{k} = m.kerns{k}.dKin(hprs{k},m.ttQuad(:,:,timeEval),Z{k});
        [~,dKzzin{k}] = m.kerns{k}.dKin(hprs{k},Z{k});
    end
end

KMats.Quad.Ktz = KtzQuad;
KMats.Quad.Ktt = KttQuad;
KMats.Obs.Ktz = KtzObs;
KMats.Kzz = Kzz;
KMats.Kzzi = Kzzi;

if flag == 1
    
    KMats.dKzzhprs = dKzzhprs;
    KMats.Obs.dKtzhprs = dKtzObshprs;
    KMats.Quad.dKtzhprs = dKtzQuadhprs;
    KMats.Quad.dKtt = dKttQuad;
    
elseif flag == 2
    
    KMats.Quad.dKtzin = dKtzQuadin;
    KMats.Obs.dKtzin = dKtzObsin;
    KMats.dKzzin = dKzzin;
    
end
