function D = diag3D(A)
% function to make diagonal matrix from 3D vector array
N = size(A,1);
M = size(A,3);
idx1 = ((1:N) + (0:(N-1)).*N)'; % indices for each slice
idx2 = (0:(M-1)).*N^2;
ddidx = bsxfun(@plus,idx1,idx2);
ddidx = ddidx(:);

if size(A,2) == 1 % vector to diagonal
    D = zeros(N,N,M);
    D(ddidx) = A(:);
else % extract diagonal elements
    D = A(ddidx);
    D = permute(reshape(D,[N,M]),[1 3 2]);
end
    