function m = Mstep(m,Kmats);

if ~ m.opts.fixed.prs
    
    m = m.EMfunctions.Mstep_Update(m,Kmats);
    
end