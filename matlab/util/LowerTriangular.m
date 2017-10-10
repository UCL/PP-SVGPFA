function ll = LowerTriangular(A)
% takes square matrix A as input and returns lower triangular part of it as a
% vector
n = size(A,1);

ll = A(tril(true(n)));