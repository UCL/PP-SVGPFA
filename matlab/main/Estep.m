function m = Estep(m);

if m.ntr > 1
    parfor nn = 1:m.ntr
        [prs{nn},objval{nn}] = Estep_singleTrial(m,nn);
    end
else
    [prs,objval] = Estep_singleTrial(m,1);
    objval = {objval};
    prs = {prs};
end

% update values in model
m.FreeEnergy = [m.FreeEnergy,sum(cell2mat(objval))];

m = updateParameters(m,prs,1);

end