function [ddk_mu_in,ddk_sig_in,ddk_mu_inObs] = grads_inducingPoints_posteriorGP_svGPFA(m,Kmats,trEval,R);

% trEval is either the trial index or 1:m.ntr to update all trials
% simultaneously
% include helper indices to keep track of changing structure depending on
% parallel or sequential updating
if length(trEval) > 1 % evalutate over all trials
    timeEval = trEval;
elseif length(trEval) == 1
    timeEval = 1;
end


for k = 1:m.dx
    
    q_mu_k = m.q_mu{k}(:,:,trEval);
    q_sigma_k = m.q_sigma{k}(:,:,trEval);
    Ak = mtimesx(Kmats.Kzzi{k},q_mu_k);
    
    dBzz = -mtimesx(Kmats.Kzzi{k},reshape(permute(reshape(mtimesx(Kmats.Kzzi{k},...
        reshape(Kmats.dKzzin{k},m.numZ(k),[],length(trEval))),...
        m.numZ(k),m.numZ(k),[]),[2 1 3]),...
        m.numZ(k),m.numZ(k)^2,[]));
    
    dBzz = reshape(dBzz,m.numZ(k),m.numZ(k),[],length(trEval));
    
    if isfield(Kmats,'Quad')
        
        % derivative of mean for quadrature term
        mm1 = reshape(mtimesx(Ak,'T',...
            reshape(permute(Kmats.Quad.dKtzin{k},[2 1 3 4]),m.numZ(k),[],m.ntr)),[],m.numZ(k),length(trEval));
        
        mm2 = mtimesx(permute(q_mu_k,[1 2 4 3]),permute(Kmats.Quad.Ktz{k},[4 2 1 3]));
        mm22 = mtimesx(reshape(mm2,[],m.opts.nquad,1,length(trEval)),'T',reshape(permute(dBzz,[2 1 5 3 4]),[],1,m.numZ(k),length(trEval)));        
        ddk_mu_in{k} = mm1 + squeeze(mm22);        
      
        
        for nn = 1:length(trEval)
            ddObsin1 = permute(mtimesx(Ak(:,:,timeEval(nn)),'T',permute(Kmats.Obs.dKtzin{k,timeEval(nn)},[2 1 3])),[2 3 1]);
            ddObsin2 = permute(mtimesx(mtimesx(Kmats.Obs.Ktz{k,timeEval(nn)},dBzz(:,:,:,timeEval(nn))),q_mu_k(:,:,timeEval(nn))),[1 3 2]);
            ggn{nn} = m.prs.C(m.index{trEval(nn)},k)'*(ddObsin1 + ddObsin2);
        end
        
        ddk_mu_inObs{k} = ggn;

    else
        mm1 = reshape(mtimesx(Ak,'T',...
            reshape(permute(Kmats.dKtzin{k},[2 1 3 4]),m.numZ(k),[],m.ntr)),[],m.numZ(k),length(trEval));
        
        mm2 = mtimesx(permute(q_mu_k,[1 2 4 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
        mm22 = mtimesx(reshape(mm2,[],R,1,length(trEval)),'T',reshape(permute(dBzz,[2 1 5 3 4]),[],1,m.numZ(k),length(trEval)));
        
        ddk_mu_in{k} = mm1 + squeeze(mm22);
    end
    
    
    
    if nargout > 1
        
        
        if isfield(Kmats,'Quad')
            mm4 = mtimesx(Kmats.Kzzi{k},Kmats.Quad.Ktz{k},'T');
            mm3 = mtimesx(q_sigma_k,mm4);
            mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Quad.Ktz{k},[4 2 1 3]));
            mm5 = reshape(mm5,[],1,m.opts.nquad,length(trEval));
            mm6 = permute(reshape(dBzz,[],m.numZ(k),length(trEval)),[2 1 4 3]);
            mm7 = mtimesx(Kmats.Kzzi{k},mm3);
            
            ddk_sig_in{k} = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.Quad.dKtzin{k},[2 3 1 4])),[3 2 4 1]) ...
                + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
                - 2*permute(mtimesx(permute(Kmats.Quad.dKtzin{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
                - permute(mtimesx(permute(Kmats.Quad.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBzz,[1 2 4 3]),Kmats.Quad.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
            
        else
            
            mm4 =  mtimesx(Kmats.Kzzi{k},Kmats.Ktz{k},'T');
            mm3 = mtimesx(q_sigma_k,mm4);
            mm5 = bsxfun(@times,permute(mm3,[1 4 2 3]),permute(Kmats.Ktz{k},[4 2 1 3]));
            mm5 = reshape(mm5,[],1,R,length(trEval));
            mm6 = permute(reshape(dBzz,[],m.numZ(k),length(trEval)),[2 1 4 3]);
            mm7 = mtimesx(Kmats.Kzzi{k},mm3);
            
            
            ddk_sig_in{k} = 2*permute(mtimesx(permute(mm7,[4 1 2 3]),permute(Kmats.dKtzin{k},[2 3 1 4])),[3 2 4 1]) ...
                + 2*permute(mtimesx(mm6,mm5),[3 1 4 2]) ...
                - 2*permute(mtimesx(permute(Kmats.dKtzin{k},[3 2 1 4]),permute(mm4,[1 4 2 3])),[3 1 4 2]) ...
                - permute(mtimesx(permute(Kmats.Ktz{k},[4 2 1 3]),permute(mtimesx(permute(dBzz,[1 2 4 3]),Kmats.Ktz{k},'T'),[1 4 2 3])),[3 2 4 1]);
            
        end
    end
    
end
