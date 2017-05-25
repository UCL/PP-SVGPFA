import tensorflow as tf
import numpy as np
from settings import float_type, jitter_level,std_qmu_init, np_float_type, np_int_type
from functions import eye, variational_expectations
from mean_functions import Zero
from kullback_leiblers import gauss_kl_white, gauss_kl_white_diag, gauss_kl, gauss_kl_diag
from conditionals import conditional
from quadrature import hermgauss


class ChainedGPs(object):
    """
    Chained Gaussian Processes

    The key reference for this algorithm is:
    ::
      @article{saul2016chained,
        title={Chained Gaussian Processes},
        author={Saul, Alan D and Hensman, James and Vehtari, Aki and Lawrence, Neil D},
        journal={arXiv preprint arXiv:1604.05263},
        year={2016}
      }
    
    """
    def __init__(self, X, Y, kerns,likelihood,Zs,mean_functions=None, whiten=True,q_diag=False, f_indices=None):
        '''
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns, likelihood, mean_functions are appropriate (single or list of) GPflow objects
        - Zs is a list of  matrices of pseudo inputs, size M[k] x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        '''
        self.likelihood = likelihood
        self.kerns = kerns
        self.D = len(kerns)
        self.mean_functions = [Zero() for _ in range(self.D)] if mean_functions is None else mean_functions
        self.f_indices = f_indices # function of one variable
        self.X = X
        self.Y = Y
        self.Zs = Zs
        self.num_inducing = [z.get_shape()[0] for z in Zs]
        self.num_latent = Y.get_shape()[-1]
        self.num_data = Y.get_shape()[0]
        self.whiten=whiten
        self.q_diag = q_diag
        self.initialize_inference()

    def initialize_inference(self):

        with tf.variable_scope("inference") as scope:
            self.q_mu,self.q_sqrt = [],[]
            for i in range(self.D):

                self.q_mu.append( tf.get_variable("q_mu%d"%i,[self.num_inducing[i], self.num_latent],\
                          initializer=tf.constant_initializer(np.random.randn(self.num_inducing[i], self.num_latent)*std_qmu_init,\
                                                              dtype=float_type)))

                if self.q_diag:
                    q_sqrt = np.ones((self.num_inducing[i], self.num_latent))
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%i,[self.num_inducing[i],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

                else:
                    q_sqrt = np.array([np.eye(self.num_inducing[i]) for _ in range(self.num_latent)]).swapaxes(0, 2)
                    self.q_sqrt.append( tf.get_variable("q_sqrt%d"%i,[self.num_inducing[i],self.num_inducing[i],self.num_latent], \
                                          initializer=tf.constant_initializer(q_sqrt,dtype=float_type)) )

    def build_prior_KL(self):

        KL = tf.Variable(0,name='KL',trainable=False,dtype=float_type)
        for i in range(self.D):
            if self.whiten:
                if self.q_diag:
                    KL += gauss_kl_white_diag(self.q_mu[i], self.q_sqrt[i])
                else:
                    KL += gauss_kl_white(self.q_mu[i], self.q_sqrt[i])
            else:
                K = self.kerns[i].K(self.Zs[self.f_indices[i]]) + eye(self.num_inducing[i]) * jitter_level
                if self.q_diag:
                    KL += gauss_kl_diag(self.q_mu[i], self.q_sqrt[i], K)
                else:
                    KL += gauss_kl(self.q_mu[i], self.q_sqrt[i], K)
        return KL

    def build_predict_fs(self, Xnew, full_cov=False):

        mus, vars = [],[]
        for i in range(self.D):
            x = tf.reshape(Xnew[:,self.f_indices[i]],[-1,1])
            mu, var = conditional(x, self.Zs[self.f_indices[i]], self.kerns[i], self.q_mu[i],
                                           q_sqrt=self.q_sqrt[i], full_cov=full_cov, whiten=self.whiten)
            mus.append(mu+self.mean_functions[i](x))
            vars.append(var)
        return tf.stack(mus),tf.stack(vars)

class VGPFA(object):
    """
    This method approximates the posterior on latent, using a mean-field variational approximation.
    Joint density over the latents is approximated as a product multivariate Gaussian densities over each latent.

    The key reference from which this algorithm is built upon is:
    ::
      @article{Opper:2009,
          title = {The Variational Gaussian Approximation Revisited},
          author = {Opper, Manfred and Archambeau, Cedric},
          journal = {Neural Comput.},
          year = {2009},
          pages = {786--792},
      }
    A secondary reference for the treatment of the additive case is
    ::
      @inproceedings{adam2016scalable,
          title={Scalable Transformed additive signal decomposition by non-conjugate gaussian process inference}
          author={Adam, Vincent, Hensman, James and Sahani Maneesh},
          booktitle={MLSP, IEEE},
          year={2016}
      }          
    """
    def __init__(self, X, Y, kerns,likelihood,C,b,units_out=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kerns, likelihood, mean_function are appropriate (lists of or single) GPflow objects
        C is a loading matrix, size O x D
        d is an offset vector, size O
        """
        self.likelihood = likelihood
        self.kerns = kerns
        self.D = len(kerns)
        self.mean_functions = [Zero() for _ in range(self.D)]
        self.f_indices = [0 for d_ in range(self.D)]
        self.X = X
        self.Y = Y
        self.num_data = Y.get_shape()[0]
        self.num_out= Y.get_shape()[1]
        self.num_latent = Y.get_shape()[2]
        self.initialize_inference()

        # mask for likelihood evaluation
        self.units_out = np.array([],dtype=np_int_type) if units_out is None else units_out
        mask = np.ones((self.num_data.value,self.num_out.value,self.num_latent.value))
        mask[:,self.units_out,:] = 0
        self.mask = tf.constant(mask,dtype=float_type)

        self.initialize_parameters(C,b)

    def initialize_parameters(self, C,b):
        with tf.variable_scope("param") as scope:
            with tf.variable_scope("loading") as scope:
                self.C= tf.get_variable("C",[self.num_out ,self.D],\
                                initializer=tf.constant_initializer(C,dtype=float_type))
            with tf.variable_scope("offset") as scope:
                self.b= tf.get_variable("b",[self.num_out],\
                                initializer=tf.constant_initializer(b,dtype=float_type))

    def initialize_inference(self):
        with tf.variable_scope("inference") as scope:
            self.q_alpha = tf.get_variable("q_alpha",[ self.D,self.num_data, self.num_latent], \
                 initializer=tf.constant_initializer(np.zeros((self.D,self.num_data, self.num_latent)),dtype=float_type))
            self.q_lambda = tf.get_variable("q_lambda",[ self.D,self.num_data, self.num_latent], \
                 initializer=tf.constant_initializer(np.ones((self.D,self.num_data, self.num_latent)),dtype=float_type))


    def build_likelihood(self):
        """
        q_alpha, q_lambda are variational parameters, size N x R
        This method computes the variational lower bound on the likelihood,
        which is:
            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]
        with
            q(f) = N(f | K alpha + mean, [K^-1 + diag(square(lambda))]^-1) .
        """
        cost = -self.build_prior_KL()
        y_mean, y_var = self.predict_log_rates(self.X)
        v_exp = self.likelihood.variational_expectations(y_mean, y_var, self.Y)
        cost += tf.reduce_sum(v_exp*self.mask)
        return cost

    def build_prior_KL(self):

        KL = tf.Variable(0,name='KL',trainable=False,dtype=float_type)

        for d in range(self.D):
            K = self.kerns[d].K(self.X)
            K_alpha = tf.matmul(K, self.q_alpha[d,:,:])
            f_mean = K_alpha + self.mean_functions[d](self.X)

            # compute the variance for each of the outputs
            I = tf.tile(tf.expand_dims(eye(self.num_data), 0), [self.num_latent.value, 1, 1])
            A = I + tf.expand_dims(tf.transpose(self.q_lambda[d,:,:]), 1) * \
                tf.expand_dims(tf.transpose(self.q_lambda[d,:,:]), 2) * K
            L = tf.cholesky(A)
            Li = tf.matrix_triangular_solve(L, I)
            tmp = Li / tf.expand_dims(tf.transpose(self.q_lambda[d,:,:]), 1)
            f_var = 1./tf.square(self.q_lambda[d,:,:]) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

            # some statistics about A are used in the KL
            A_logdet = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
            trAi = tf.reduce_sum(tf.square(Li))

            KL += 0.5 * (A_logdet + trAi - self.num_data.value * self.num_latent.value +
                        tf.reduce_sum(K_alpha*self.q_alpha[d,:,:]))
        return KL

    def build_predict_fs(self, Xnew, full_cov=False):
        """
        The posterior variance of F is given by
            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)
        Here we project this to F*, the values of the GP at Xnew which is given
        by
           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )
        """
        f_means, f_vars = [],[]
        for d in range(self.D):
            # compute kernel things
            Kx = self.kerns[d].K(self.X, Xnew)
            K = self.kerns[d].K(self.X)

            # predictive mean
            f_mean = tf.matmul(Kx, self.q_alpha[d,:,:], transpose_a=True) + self.mean_functions[d](Xnew)

            # predictive var
            A = K + tf.matrix_diag(tf.transpose(1./tf.square(self.q_lambda[d,:,:])))
            L = tf.cholesky(A)
            Kx_tiled = tf.tile(tf.expand_dims(Kx, 0), [self.num_latent.value, 1, 1])
            LiKx = tf.matrix_triangular_solve(L, Kx_tiled)
            if full_cov:
                f_var = self.kerns[d].K(Xnew) - tf.matmul(LiKx, LiKx, transpose_a=True)
            else:
                f_var = self.kerns[d].Kdiag(Xnew) - tf.reduce_sum(tf.square(LiKx), 1)
            f_means.append(f_mean)
            f_vars.append(tf.transpose(f_var))

        return tf.stack(f_means),tf.stack(f_vars)

    def predict_log_rates(self,Xnew):

        # predicting latent functions
        fmeans, fvars = self.build_predict_fs(Xnew, full_cov=False)

        # linear expansion
        return tf.einsum('od,dnr->nor',self.C,fmeans) + tf.expand_dims(tf.expand_dims(self.b,0),2),\
               tf.einsum('od,dnr->nor',tf.square(self.C),fvars)

class SVGPFA(ChainedGPs):
    """
    Sparse Variational inference for GPFA
    """

    def __init__(self, X, Y, kerns,likelihood,Zs,C,b, whiten=True,q_diag=True,units_out=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kerns, likelihood, mean_function are appropriate (lists of or single) GPflow objects
        - Zs is a list of matrices of pseudo inputs, size M[k] x D
        - C is a loading matrix, size O x D
        - d is an offset vector, size O
        """
        self.likelihood = likelihood
        self.kerns = kerns
        self.D = len(kerns)
        self.mean_functions = [Zero() for _ in range(self.D)]
        self.X = X
        self.Y = Y
        self.Zs = Zs
        self.num_inducing = [z.get_shape()[0] for z in Zs]
        self.whiten=whiten
        self.q_diag = q_diag
        self.f_indices = [0 for _ in range(self.D)] # shared covariate
        self.num_data = Y.get_shape()[0]
        self.num_out = Y.get_shape()[1]
        self.num_latent = Y.get_shape()[2]
        self.initialize_inference()

        # mask for likelihood evaluation
        self.units_out = np.array([],dtype=np_int_type) if units_out is None else units_out
        mask = np.ones((self.num_data.value,self.num_out.value,self.num_latent.value))
        mask[:,self.units_out,:] = 0
        self.mask = tf.constant(mask,dtype=float_type)

        with tf.variable_scope("param") as scope:
            self.C= tf.get_variable("C",[self.num_out ,self.D],initializer=tf.constant_initializer(C,dtype=float_type))
            self.b= tf.get_variable("b",[self.num_out],initializer=tf.constant_initializer(b,dtype=float_type))

    def build_likelihood(self):

        cost = -self.build_prior_KL()

        # predicting predictor
        pred_mean,pred_var = self.predict_log_rates(self.X)
        # variational expectations
        var_exp = self.likelihood.variational_expectations(pred_mean,
                                                           pred_var, self.Y)
        cost += tf.reduce_sum(var_exp*self.mask)

        return cost

    def predict_log_rates(self,Xnew):
        # predicting latent functions
        fmeans, fvars = self.build_predict_fs(Xnew, full_cov=False)
        # linear expansion
        return tf.einsum('od,dnr->nor',self.C,fmeans) + tf.expand_dims(tf.expand_dims(self.b,0),-1) ,\
               tf.einsum('od,dnr->nor',tf.square(self.C),fvars)

class PP_SVGPFA(ChainedGPs):
    """
    Point Process Sparse Variational inference 
    """

    def __init__(self,T, X,mask, kerns,Zs,C,b, whiten=True,q_diag=True,n_grid=1000,link=tf.exp):
        """
        - T is a scalar, the time horizon
        - X is a data matrix, size N x R
        - mask is a matrix, size N x O X R
        - kerns, mean_function are appropriate (lists of or single) GPflow objects
        - Zs is a list of matrices of pseudo inputs, size M[k] x D
        - C is a loading matrix, size O x D
        - d is an offset vector, size O
        - link is a positive tensorflow function
        """

        self.kerns = kerns
        self.D = len(kerns)
        self.mean_functions = [Zero() for _ in range(self.D)]

        self.num_out = mask.get_shape()[1]
        self.N = mask.get_shape()[0]
        self.num_latent = mask.get_shape()[2]

        self.T=T
        self.X=X
        self.mask = mask

        self.n_grid = n_grid
        self.link = link
        self.T_grid =tf.convert_to_tensor(np.linspace(0,T,n_grid).reshape(-1,1),dtype=float_type)

        self.Zs = Zs
        self.num_inducing = [z.get_shape()[0] for z in Zs]
        self.num_inducing_np = [z.get_shape().as_list()[0] for z in Zs]
        self.whiten=whiten
        self.q_diag = q_diag
        self.f_indices=[0 for _ in range(self.D)]

        self.initialize_inference()

        with tf.variable_scope("param") as scope:
            self.C= tf.get_variable("C",[self.num_out ,self.D],initializer=tf.constant_initializer(C,dtype=float_type))
            self.b= tf.get_variable("b",[self.num_out],initializer=tf.constant_initializer(b,dtype=float_type))


    def build_likelihood(self):
        cost = -self.build_prior_KL()
        # variational expectations
        if self.link == tf.exp:
            pred_mean,pred_var = self.predict_log_rates(self.X)
            cost += tf.reduce_sum( pred_mean*self.mask  )
        else:
            pred_mean,pred_var = self.predict_log_rates(self.X)
            vexp = variational_expectations(pred_mean,pred_var, lambda x: tf.log(self.link(x)))
            cost += tf.reduce_sum( vexp*self.mask )
            # penalty (fast prediction of all trials and output on shared time vector)
        cost -= self.build_normalizer_grid()
        return cost


    def build_normalizer_adaptive(self):
        num_gauss_hermite_points = 10
        gh_x, _ = hermgauss(num_gauss_hermite_points)
        gh_x = gh_x.reshape(1, -1)
        Z_int = tf.reshape(tf.stack([ z + gh_x*self.kerns[d].lengthscales for d,z in enumerate(self.Zs) ]),[-1])
        Z_int = tf.reshape(tf.gather(Z_int, tf.nn.top_k(Z_int, k=Z_int.get_shape().as_list()[0]).indices),[-1,1])
        n_grid = Z_int.get_shape().as_list()[0]
        W_int = tf.ones_like(Z_int)*self.T/n_grid
        # reshape for output
        if self.link == tf.exp:
            pred_mean,pred_var = self.predict_log_rates(Z_int)
            W_int = tf.zeros_like(pred_mean) + tf.expand_dims(W_int,0)
            return tf.reduce_sum( tf.exp( pred_mean + .5*tf.sqrt(pred_var) )*W_int )

    def build_normalizer_grid(self):
        t_grid = tf.reshape(np.linspace(0,self.T,self.n_grid,dtype=np_float_type),[-1,1])
        w_grid = tf.ones_like(t_grid)*self.T/self.n_grid
        pred_mean,pred_var = self.predict_log_rates(t_grid)
        w_grid = tf.zeros_like(pred_mean) + tf.expand_dims(w_grid,-1)

        # reshape for output
        if self.link == tf.exp:
            return tf.reduce_sum( tf.exp( pred_mean + .5*tf.sqrt(pred_var) )*w_grid )
        else:
            vexp = variational_expectations(pred_mean,pred_var, lambda x: self.link(x)  )
            return tf.reduce_sum( vexp*w_grid )

    def predict_log_rates(self,Xnew):
        # predicting latent functions
        fmeans, fvars = self.build_predict_fs(Xnew, full_cov=False)
        # linear expansion
        return tf.einsum('od,dnr->nor',self.C,fmeans)+ tf.expand_dims(tf.expand_dims(self.b,0),-1),\
               tf.einsum('od,dnr->nor',tf.square(self.C),fvars)

