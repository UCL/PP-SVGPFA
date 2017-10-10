function KMats = KernelMatrices_prediction_svGPFA(m,testTimes);

Kzz = cell(m.dx,1);
Kzzi = cell(m.dx,1);
Ktz = cell(m.dx,1);
Ktt = zeros(length(testTimes),m.dx);


for k = 1:m.dx

    Ktt(:,k) = m.kerns{k}.Kdiag(m.kerns{k}.hprs,testTimes);

    
        Kzz{k} = m.kerns{k}.K(m.kerns{k}.hprs,m.Z{k}) + m.epsilon*eye(m.numZ(k));
        for numtr = 1:m.ntr
            Kzzi{k}(:,:,numtr) = pinv(Kzz{k}(:,:,numtr));
        end
        Ktz{k}= m.kerns{k}.K(m.kerns{k}.hprs,testTimes,m.Z{k});
    
end

KMats.Kzz = Kzz;
KMats.Kzzi = Kzzi;
KMats.Ktz = Ktz;
KMats.Ktt = Ktt;
