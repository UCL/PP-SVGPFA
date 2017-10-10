function x = logdet(A)
% LOGDET - computes the log-determinant of a matrix A
%
% x = logdet(A);
%
% This is faster and more stable than using log(det(A))
%
% Input:
%     A NxN - A must be sqaure, positive semi-definite


[C,p] = chol(A);
if p == 0
    x = 2*sum(log(diag(C)));
else
    [L, U, P] = lu(A);
    du = diag(U);
%     c = det(P) * prod(sign(du));
%     x = log(c) + sum(log(abs(du)));
    x = sum(log(abs(du))); % avoid complex numbers du

end