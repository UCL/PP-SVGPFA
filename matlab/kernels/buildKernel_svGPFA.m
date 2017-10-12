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
        
        kern.numhprs = 4;
        kern.hprs = hprs;
        kern.K = @LocallyPeriodicKernel;
        kern.Kdiag = @Kdiag_LocallyPeriodicKernel;
        kern.dKhprs = @dKhprs_LocallyPeriodicKernel;
        kern.dKin = @dKin_LocallyPeriodicKernel;
        
    case 'Matern32'
        
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = @matern32Kernel;
        kern.Kdiag = @Kdiag_matern32Kernel;
        kern.dKhprs = @dKhprs_matern32Kernel;
        kern.dKin = @dKin_matern32Kernel;
        
    case 'Matern52'
        
        kern.numhprs = 2;
        kern.hprs = hprs;
        kern.K = @matern52Kernel;
        kern.Kdiag = @Kdiag_matern52Kernel;
        kern.dKhprs = @dKhprs_matern52Kernel;
        kern.dKin = @dKin_matern52Kernel;
        
    case 'RationalQuadratic'
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = @RationalQuadraticKernel;
        kern.Kdiag = @Kdiag_RationalQuadraticKernel;
        kern.dKhprs = @dKhprs_RationalQuadraticKernel;
        kern.dKin = @dKin_RationalQuadraticKernel;
        
    case 'Linear'
        
        error('not implemented yet')
        
        kern.numhprs = 3;
        kern.hprs = hprs;
        kern.K = str2func('LinearKernel');
end