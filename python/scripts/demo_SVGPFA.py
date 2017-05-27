"""

This script demonstrates the use of multiple algorithms to run learning and inference for GPFA using binned data.

Algorithms are:
- VGPFA: Variational Gaussian Process Factor Analysis
- SVGPFA: Sparse Variational Gaussian Process Factor Analysis

The framework is flexible and arbitrary likelihood may be used (Gaussian, Poisson...)

"""

import sys
sys.path.append('../pp-SVGPFA')

import tensorflow as tf
import numpy as np
from kernels import RBF
from matplotlib import pyplot as plt
from matplotlib import cm
from likelihoods import Gaussian, Poisson, Gaussian_with_link
from settings import np_float_type,int_type
from model import SVGPFA,VGPFA
from simulate_point_process import simulate_GPFA_rates_from_latent
from tensorflow.contrib.opt import ScipyOptimizerInterface as soi
np.random.seed(10)


model = 'SVGPFA'
lik = 'Poisson'

assert model in ['VGPFA','SVGPFA']
assert lik in ['Poisson', 'Gaussian', 'Gaussian_link']

print('model:%s,likelihood:%s'%(model,lik))

#---------------------------------------------------
# Declaring GPFA model parameters

D = 2 # number of additive terms
R = 1 # number of trials
O = 50 # number of neurons
C0 = np.random.randn(O,D)*1.
d0 = np.random.randn(O,1)*1.
fs = [lambda x:np.sin(3*x), lambda x:np.cos(x)]
T = 10.

#---------------------------------------------------
# Simulating data

np_link,tf_link = np.exp, tf.exp
bin_width=0.02
X_np, Rates_np, Preds_np = simulate_GPFA_rates_from_latent(fs, C0, d0, T, R=R, bin_width=bin_width ,link=np_link)
N = X_np.shape[0]
if lik == 'Poisson':
    Y_np = np.random.poisson(Rates_np)
else:
    Y_np = Rates_np + np.random.randn(N, O, R)*.1
print(X_np.shape, Y_np.shape, Preds_np.shape)


plt.imshow(Rates_np[:,:,0].T,interpolation='nearest',origin='lower',aspect='auto',cmap=cm.gray)
plt.colorbar()
plt.title('binned raster plot')
plt.xlabel('time')
plt.ylabel('neuron index')
plt.savefig('%s_sim.pdf'%model)
plt.close()

#---------------------------------------------------
# Constructing tensorflow model

X = tf.placeholder(tf.float32,[N,1])
Y = tf.placeholder(tf.float32,[N,O,R])
ks,Zs = [],[]
with tf.variable_scope("kernels") as scope:
    for d_ in range(D):
        with tf.variable_scope("kernel%d"%d_) as scope:
            ks.append(  RBF(1,lengthscales=.5,variance=1.) )
with tf.variable_scope("likelihood") as scope:
    if lik == 'Poisson':
        likelihood = Poisson(invlink=tf_link)
    elif lik == 'Gaussian':
        likelihood = Gaussian(variance=.1)
    elif lik == 'Gaussian_link':
        likelihood = Gaussian_with_link(variance=.1, invlink=tf_link)
Nz = [20 for _ in range(D)] # inducing points
Zs_np = [np.random.uniform(0,T,Nz[d_]).astype(np_float_type).reshape(-1,1) for d_ in range(D)]
for d_ in range(D):
    with tf.variable_scope("ind_points%d"%d_) as scope:
        Zs.append(   tf.Variable(Zs_np[d_],\
                     tf.float32,name='Z') )

with tf.variable_scope("model") as scope:
    if model == 'SVGPFA':
        m= SVGPFA(X,Y,ks,likelihood,Zs,C0,d0,q_diag=False)
    elif model == 'VGPFA':
        m= VGPFA(X,Y,ks,likelihood,C0,d0)

#---------------------------------------------------
sess  = tf.Session()
sess.run(tf.global_variables_initializer()) # reset values to wrong

# declare loss
loss = -m.build_likelihood()
# separate variables
vars_e, vars_m, vars_h= [], [], []
vars_e += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/inference')
vars_m += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/param')
vars_m += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='likelihood')
vars_h += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='kernels')
# declare optimizers
opt_e = soi(loss, var_list=vars_e, method='L-BFGS-B', options={'ftol': 1e-3})
opt_m = soi(loss, var_list=vars_m, method='L-BFGS-B', options={'ftol': 1e-2})
opt_h = soi(loss, var_list=vars_h, method='L-BFGS-B', options={'ftol': 1e-2})

init = tf.global_variables_initializer()
sess.run(init) # reset values to wrong
feed_dic = {X:X_np, Y:Y_np}

#---------------------------------------------------

colors_o = plt.cm.jet(np.linspace(0,1,O))
colors_d = plt.cm.winter(np.linspace(0,1,D))


print('Optimized variables:')
for var in vars_e+vars_m+vars_h:
    print(var.name)  # Prints the name of the variable alongside its val

nit = 500
loss_array = np.zeros((nit,))
x = X_np[:,0]

print('Starting Optimization')

print('1rst E_step')
opt_e.minimize(sess, feed_dict=feed_dic)
print('1rst H_step')
opt_h.minimize(sess, feed_dict=feed_dic)

for i in range(1,nit):
    print('E_step %d/%d '%(i,nit))
    opt_e.minimize(sess, feed_dict=feed_dic)
    print('M_step %d/%d '%(i,nit))
    opt_m.minimize(sess, feed_dict=feed_dic)
    if i%5==0:
        print('H_step %d/%d '%(i,nit))
        opt_h.minimize(sess, feed_dict=feed_dic)

    loss_array[i]= float(sess.run(loss, feed_dic))

    if i%10 == 0:
        Fs_mean,Fs_var = sess.run(m.build_predict_fs(X),  feed_dic)
        y_mean,y_var = sess.run(m.predict_log_rates(X),  feed_dic)
        for d in range(D):
            for r in range(R):
                f, s = Fs_mean[d, :, r], np.sqrt(Fs_var[d, :, r])
                plt.plot(x,f,color=colors_d[d])
                plt.plot(x,fs[d](x),color=colors_d[d])
                plt.fill_between(x.flatten(), f - s, y2=f + s, alpha=.3, facecolor=colors_d[d])
        plt.xlabel('time (s)')
        plt.title('true and inferred latents')
        plt.savefig('%s_predict_latent.pdf' % model)
        plt.close()
        for o in range(O):
            for r in range(R):
                y = y_mean[:,o,r]
                plt.plot(Preds_np[:,o,r].flatten(),y.flatten(),color=colors_o[o])
        plt.xlabel('log rate (true)')
        plt.xlabel('log rate (predicted)')
        plt.savefig('%s_log_rates.pdf' % model)

        plt.close()


plt.plot(loss_array)
plt.savefig('svgpfa_add_loss.pdf')
plt.close()

#=================================




