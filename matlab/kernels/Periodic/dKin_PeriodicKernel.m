function [dGin2, dGin1] = dKin_PeriodicKernel(prs,X1,varargin)

variance = prs(1);
lengthscale = prs(2);
period = prs(3);

% take care of empty input
if isempty(X1)
    X1 = zeros(0,1);
end

% inputs
if nargin == 2
    X2 = X1;
else
    X2 = varargin{1};
    if isempty(X2)
        X2 = zeros(0,1);
    end
end

G = PeriodicKernel(prs,X1,X2);
dd = bsxfun(@minus,X1,X2');
rr = (pi.*dd./period);

dGin2 = zeros([size(G),length(X2)]);

if nargout > 1
    dGin1 = zeros([size(G),length(X1)]);
end

for ii = 1:length(X2);
    dGin2(:,ii,ii) = 4*pi/period*G(:,ii)/lengthscale^2 .* ...
        sin(rr(:,ii)).*cos(rr(:,ii));
end


% gradient with respect to input points X2
if nargout > 1
    for ii = 1:length(X1)
        if nargin == 2 % grad wrt to first input if same
            dGin1(ii,:,ii) = -4*pi/period*G(ii,:)/lengthscale^2 .* ...
                sin(rr(ii,:)).*cos(rr(ii,:));
            dGin1(:,ii,ii) = 4*pi/period*G(:,ii)/lengthscale^2 .* ...
                sin(rr(:,ii)).*cos(rr(:,ii));
        else % grad wrt first input if different
            dGin1(ii,:,ii) = -4*pi/period*G(ii,:)/lengthscale^2 .* ...
                sin(rr(ii,:)).*cos(rr(ii,:));
        end
    end
end



