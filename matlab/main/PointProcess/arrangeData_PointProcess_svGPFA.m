function [Ystack,index] = arrangeData_PointProcess_svGPFA(m,Y);

% stack all spike times into one big vector for each trial
Ystack = cellfun(@(x)cellvec(x),Y,'uni',0);

% get indices to remember which spike belongs to which neuron
index = cell(m.ntr,1);
for numtr = 1:m.ntr % indices to be used when concatenating all spikes for each trial
    index{numtr} = cellvec(cellfun(@(x,y)repmat(x,[y 1]),num2cell(1:m.dy),(cellfun(@length,Y{numtr},'uni',0))','uni',0));
end

