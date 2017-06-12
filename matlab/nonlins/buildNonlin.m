function nonlin = buildNonlin(name,varargin)

switch name
    case 'Exponential'
        nonlin.name = 'Exponential';
        nonlin.Exponential = @(x) exp(x);
    case 'SoftRec'
        nonlin.name = 'SoftRec';
        nonlin.hprs = varargin{1};
        nonlin.SoftRec = @(X,alpha) 1/alpha * log(1 + exp(alpha.*X));
end
