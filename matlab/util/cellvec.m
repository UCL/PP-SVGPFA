function v = cellvec(x)
% CELLVEC - vectorizes cell input
% v = cellvec(x)
% concatenates vectors in cell array into a single long vector
v = vertcat(x{:});