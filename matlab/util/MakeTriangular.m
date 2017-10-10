function A = MakeTriangular(ll)
% takes square matrix A as input and returns lower triangular part of it as a
% vector
n = sqrt(2*length(ll) + 1/4) - 1/2;
A = zeros(n);
A(tril(true(n))) = ll;