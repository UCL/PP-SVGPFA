function m = setFunctionHandles_svGPFA(m,lik);
% function to set the function handles of the model

switch lik
    case 'Gaussian'
        m.EMfunctions.BuildKernelMatrices = @BuildKernelMatrices_Gaussian_svGPFA;
        m.EMfunctions.likelihood = @Lik_Gaussian_svGPFA;
        m.EMfunctions.gradLik_hprs = @gradLik_hprs_Gaussian_svGPFA;
        m.EMfunctions.gradLik_inducingPoints = @gradLik_inducingPoints_Gaussian_svGPFA;
        m.EMfunctions.Estep_Update = @Estep_Update_Gaussian;
        m.EMfunctions.Mstep_Update = @Mstep_Update_Gaussian;

    case 'Poisson'
        m.EMfunctions.BuildKernelMatrices = @BuildKernelMatrices_Poisson_svGPFA;
        m.EMfunctions.likelihood = @Lik_Poisson_svGPFA;
        m.EMfunctions.gradLik_hprs = @gradLik_hprs_Poisson_svGPFA;
        m.EMfunctions.gradLik_inducingPoints = @gradLik_inducingPoints_Poisson_svGPFA;
        m.EMfunctions.gradLik_variationalPrs = @gradLik_VariationalPrs_Poisson_svGPFA;
        m.EMfunctions.gradLik_modelPrs = @gradLik_ModelPrs_Poisson_svGPFA;         
        m.EMfunctions.Estep_Update = @Estep_Update_Iterative_svGPFA;
        m.EMfunctions.Mstep_Update = @Mstep_Update_Iterative_svGPFA;
        
    case 'PointProcess'
        m.EMfunctions.BuildKernelMatrices = @BuildKernelMatrices_PointProcess_svGPFA;
        m.EMfunctions.likelihood = @Lik_PointProcess_svGPFA;
        m.EMfunctions.gradLik_hprs = @gradLik_hprs_PointProcess_svGPFA;
        m.EMfunctions.gradLik_inducingPoints = @gradLik_inducingPoints_PointProcess_svGPFA;
        m.EMfunctions.gradLik_variationalPrs = @gradLik_VariationalPrs_PointProcess_svGPFA;
        m.EMfunctions.gradLik_modelPrs = @gradLik_ModelPrs_PointProcess_svGPFA;
        m.EMfunctions.Estep_Update = @Estep_Update_Iterative_svGPFA;
        m.EMfunctions.Mstep_Update = @Mstep_Update_Iterative_svGPFA;
end

