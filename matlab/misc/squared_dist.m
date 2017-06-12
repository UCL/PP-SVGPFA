function dist = squared_dist(X1,varargin)
% computes matrix of squared distance between input vector and itself, or
% two input vectors
if nargin == 1
    X2 = X1;
else
    X2 = varargin{1};
end
dist = bsxfun(@minus,X1,X2').^2;
