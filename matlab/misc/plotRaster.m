function plotRaster(sts, color)
% plotRaster(sts, color);
% funtion to draw raster plot

if ~iscell(sts)
    sts = {sts};
end

len = 1:length(sts);

if nargin < 2
    color = [0 0 0];
end

for k = 1:length(sts)
    st = sts{k}(:)';
    line([st; st], [len(k); len(k)+1]*ones(size(st)) + [0.1; -0.1]*ones(size(st)), 'Color', color);
end

set(gca, 'Box', 'on');
ylim([1, max(len)+1]);
