function [dGin2, dGin1] = dKin_LocallyPeriodicKernel(prs,X1,varargin)

% hyperparameters
variance = prs(1);
lengthscale_se = prs(2);
lengthscale_per = prs(3);
period_per = prs(4);

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

G1 = rbfKernel([variance lengthscale_se],X1,X2);
G2 = PeriodicKernel([1 lengthscale_per period_per],X1,X2);

dd = bsxfun(@minus,X1,permute(X2,[2 1 3]));
rr = (pi.*dd./period_per);

[N1,N2,ntr] = size(G1);

dGin2 = zeros(N1,N2,N2,ntr);

if nargout > 1
    dGin1 = zeros(N1,N2,N1,ntr);
end

for ii = 1:size(X2,1)
    dGin2(:,ii,ii,:) = permute(4*pi/period_per*G1(:,ii,:).*G2(:,ii,:)/lengthscale_per^2 .* ...
        sin(rr(:,ii,:)).*cos(rr(:,ii,:)),[1 2 4 3]) ...
        + permute(G1(:,ii,:).*G2(:,ii,:)./lengthscale_se^2 .* dd(:,ii,:),[1 2 4 3]);
end


% gradient with respect to input points X2
if nargout > 1
    for ii = 1:length(X1)
        if nargin == 2 % grad wrt to first input if same
            dGin1(ii,:,ii,:) = permute(-4*pi/period_per*G1(ii,:,:).*G2(ii,:,:)/lengthscale_per^2 .* ...
                sin(rr(ii,:,:)).*cos(rr(ii,:,:)),[1 2 4 3])...
                + permute( -G1(ii,:,:).*G2(ii,:,:)./lengthscale_se^2 .* dd(ii,:,:),[1 2 4 3]);
            dGin1(:,ii,ii,:) = permute(4*pi/period_per*G1(:,ii,:).*G2(:,ii,:)/lengthscale_per^2 .* ...
                sin(rr(:,ii,:)).*cos(rr(:,ii,:)),[1 2 4 3])...
                + permute(G1(:,ii,:).*G2(:,ii,:)./lengthscale_se^2 .* dd(:,ii,:),[1 2 4 3]);
        else % grad wrt first input if different
            dGin1(ii,:,ii,:) = permute(-4*pi/period_per*G1(ii,:,:).*G2(ii,:,:)/lengthscale_per^2 .* ...
                sin(rr(ii,:,:)).*cos(rr(ii,:,:)),[1 2 4 3])...
                - permute( G1(ii,:,:).*G2(ii,:,:)./lengthscale_se^2 .* dd(ii,:,:),[1 2 4 3]);
        end
    end
end




