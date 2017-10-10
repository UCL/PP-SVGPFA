function kern = buildKernel_svGPFA(name,hprs)
% helper function to build convenient input structure for kernels to ppGPFA
% framework -- extend this with new kernel functions later

switch name
    case 'RBF'
        
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = @rbfKernel;
        kern.Kdiag = @Kdiag_rbfKernel;
        kern.dKhprs = @dKhprs_rbfKernel;
        kern.dKin = @dKin_rbfKernel;
        
    case 'Periodic'
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = @PeriodicKernel;
        kern.Kdiag = @Kdiag_PeriodicKernel;
        kern.dKhprs = @dKhprs_PeriodicKernel;
        kern.dKin = @dKin_PeriodicKernel;
        
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