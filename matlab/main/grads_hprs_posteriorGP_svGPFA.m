function [ddk_mu_hprs,ddk_sig_hprs,ddk_mu_hprsObs] = grads_hprs_posteriorGP_svGPFA(m,Kmats,R);

for k = 1:m.dx
    
    Ak = mtimesx(Kmats.Kzzi{k},m.q_mu{k});
    
    dBhh = -mtimesx(Kmats.Kzzi{k},...
        reshape(permute(reshape(...
        mtimesx(Kmats.Kzzi{k},...
        reshape(Kmats.dKzzhprs{k},m.numZ(k),[],m.ntr)),...
        m.numZ(k),m.numZ(k),[]),[2 1 3]),...
        m.numZ(k), m.kerns{k}.numhprs*m.numZ(k),[]));
    
    dBhh = reshape(dBhh,m.numZ(k),m.numZ(k),[],m.ntr);
    
    if isfield(Kmats,'Quad')
        
        % derivative of mean for quadrature term
        mm1 = reshape(...
            mtimesx(Ak,'T',...
            reshape(permute(Kmats.Quad.dKtzhprs{k},[2 1 3 4]),m.numZ(k),[],m.ntr)),[],m.kerns{k}.numhprs,m.ntr);
        
        mm2 = mtimesx(permute(m.q_mu{k},[1 2 4 3]),permute(Kmats.Quad.Ktz{k},[4 2 1 3]));
        
        mm22 = mtimesx(reshape(mm2,[],m.opts.nquad,1,m.ntr),'T',reshape(permute(dBhh,[2 1 5 3 4]),[],1,m.kerns{k}.numhprs,m.ntr));
        
        ddk_mu_hprs{k} = mm1 + squeeze(mm22);
        
        for numtr = 1:m.ntr
            ddObshprs1 = permute(mtimesx(Ak(:,:,numtr),'T',permute(Kmats.Obs.dKtzhprs{k,numtr},[2 1 3])),[2 3 1]);
            ddObshprs2 = permute(mtimesx(mtimesx(Kmats.Obs.Ktz{k,numtr},dBhh(:,:,:,numtr)),m.q_mu{k}(:,:,numtr)),[1 3 2]);
            ggn{numtr} = m.prs.C(m.index{numtr},k)'*(ddObshprs1 + ddObshprs2);
        end
        ddk_mu_hprsObs{k} = ggn;
    else
        mm1 = reshape(...
            mtimesx(Ak,'T',...
            reshape(permute(Kmats.dKtzhprs{k},[2 1 3 4]),m.numZ(k),[],m.ntr)),[],m.kerns{k}.numhprs,m.ntr);
        
        mm2 = mtimesx(permute(m.q_mu{k},[1 2 4 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
        
        mm22 = mtimesx(reshape(mm2,[],R,1,m.ntr),'T',reshape(permute(dBhh,[2 1 5 3 4]),[],1,m.kerns{k}.numhprs,m.ntr));
        
        ddk_mu_hprs{k} = mm1 + squeeze(mm22);
    end
    
    if nargout > 1

        if isfield(Kmats,'Quad')
            mm4 = mtimesx(Kmats.Kzzi{k},Kmats.Quad.Ktz{k},'T');
            mm3 = mtimesx(m.q_sigma{k},mm4);
            mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Quad.Ktz{k},[4 2 1 3]));
            mm5 = reshape(mm5,[],1,m.opts.nquad,m.ntr);
            mm6 = permute(reshape(dBhh,[],m.kerns{k}.numhprs,m.ntr),[2 1 4 3]);
            mm7 = mtimesx(Kmats.Kzzi{k},mm3);
            
            ttm1 = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.Quad.dKtzhprs{k},[2 3 1 4])),[3 2 4 1]) ...
                + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
                - 2*permute(mtimesx(permute(Kmats.Quad.dKtzhprs{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
                - permute(mtimesx(permute(Kmats.Quad.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBhh,[1 2 4 3]),Kmats.Quad.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
            
            ddk_sig_hprs{k} = bsxfun(@plus,permute(Kmats.Quad.dKtt{k},[1 3 2]),ttm1);
        else
            mm4 = mtimesx(Kmats.Kzzi{k},Kmats.Ktz{k},'T');
            mm3 = mtimesx(m.q_sigma{k},mm4);
            mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
            mm5 = reshape(mm5,[],1,R,m.ntr);
            mm6 = permute(reshape(dBhh,[],m.kerns{k}.numhprs,m.ntr),[2 1 4 3]);
            mm7 = mtimesx(Kmats.Kzzi{k},mm3);
            
            ttm1 = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.dKtzhprs{k},[2 3 1 4])),[3 2 4 1]) ...
                + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
                - 2*permute(mtimesx(permute(Kmats.dKtzhprs{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
                - permute(mtimesx(permute(Kmats.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBhh,[1 2 4 3]),Kmats.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
            
            ddk_sig_hprs{k} = bsxfun(@plus,permute(Kmats.dKtt{k},[1 3 2]),ttm1);
        end
    end
end
