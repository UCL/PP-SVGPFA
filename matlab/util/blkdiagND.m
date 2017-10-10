function y = blkdiagND(varargin)
%BLKDIAG  Block diagonal concatenation of ND matrix input arguments.
%
%                                   |A(:,:,i,j) 0 .. 0|
%   Y = BLKDIAG(A,B,...)  produces  |0 B(:,:,i,j) .. 0| along each 2D slice.
%                                   |0 0 ..     ..  ..|
%
% Lea Duncker, 2017

[p2,m2,gg,kk] = cellfun(@size,varargin);
if sum(gg)/nargin ~= gg(1)
    error('dimensionality mismatch')
end
%Precompute cumulative matrix sizes
p1 = [0, cumsum(p2)];
m1 = [0, cumsum(m2)];

y = zeros(p1(end),m1(end),gg(1),kk(1)); %Preallocate
for k=1:nargin
    y(p1(k)+1:p1(k+1),m1(k)+1:m1(k+1),:,:) = varargin{k};
end
