function kern = buildKernel(name,hprs)
% helper function to build convenient input structure for kernels to ppGPFA
% framework -- extend this with new kernel functions later

switch name
    case 'RBF'
        
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = str2func('rbfKernel');
        kern.Kdiag = str2func('Kdiag_rbfKernel');
        kern.dKhprs = str2func('dKhprs_rbfKernel');
        kern.dKin = str2func('dKin_rbfKernel');
        
    case 'Periodic'
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('PeriodicKernel');
        kern.Kdiag = str2func('Kdiag_PeriodicKernel');
        kern.dKhprs = str2func('dKhprs_PeriodicKernel');
        kern.dKin = str2func('dKin_PeriodicKernel');
        
    case 'LocallyPeriodic'
        
        error('not implemented yet')
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('LocallyPeriodicKernel');
        
    case 'Matern'
        
        error('not implemented yet')
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('MaternKernel');
        
    case 'RationalQuadratic'
        
        error('not implemented yet')
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('RationalQuadraticKernel');
    case 'Linear'
        
        error('not implemented yet')
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('LinearKernel');
end