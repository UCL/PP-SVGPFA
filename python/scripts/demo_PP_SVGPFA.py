"""
A GPFA model with point process observations is simulated.
Inference and learning are performed using pp-SVGPFA algorithm

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
from model import PP_SVGPFA
from simulate_point_process import simulate_pp_GPFA_from_latent, spikes_in_mat

from tensorflow.contrib.opt import ScipyOptimizerInterface as soi
np.random.seed(10)

model = 'pp-SVGPFA'

print('model:%s'%(model))

#---------------------------------------------------
# Declaring GPFA model parameters

D = 2 # number of additive terms
R = 1 # number of trials
O = 50 # number of neurons
C0 = np.random.randn(O,D)*1
d0 = np.random.rand(O,1)
fs = [lambda x:np.sin(x)**3, lambda x:np.cos(3.*x)]
T = 20.

#---------------------------------------------------
# Simulating data
np_link,tf_link = np.exp, tf.exp

Ts,log_rates = simulate_pp_GPFA_from_latent(fs,C0,d0,T=T,R=R,link=np_link)
# rearranging data
Ts_mat,mask_mat =  spikes_in_mat(Ts)
N = Ts_mat.shape[0]
colors_o = plt.cm.jet(np.linspace(0,1,O))
colors_d = plt.cm.winter(np.linspace(0,1,D))
for r,Ts_ in enumerate(Ts):
    for o,t in enumerate(Ts_):
        plt.plot(t,t*0.+o*R+r,linestyle=' ',marker='.',color=colors_o[o])
plt.title('raster plot')
plt.xlabel('time')
plt.ylabel('neuron index')
plt.savefig('%s_sim.pdf'%model)
plt.close()


#---------------------------------------------------
# Constructing tensorflow model

Y = tf.placeholder(tf.float32,[N,R])
mask = tf.placeholder(tf.float32,[N,O,R])
ks,Zs = [],[]
with tf.variable_scope("kernels") as scope:
    for d_ in range(D):
        with tf.variable_scope("kernel%d"%d_) as scope:
            ks.append(  RBF(1,lengthscales=.5,variance=1.) )
Nz = [30 for _ in range(D)] # inducing points
Zs_np = [np.random.uniform(0,T,Nz[d_]).astype(np_float_type).reshape(-1,1) for d_ in range(D)]
with tf.variable_scope("ind_points") as scope:
    for d_ in range(D):
        with tf.variable_scope("ind_points%d"%d_) as scope:
            Zs.append(   tf.Variable(Zs_np[d_],\
                         tf.float32,name='Z') )
with tf.variable_scope("model") as scope:
    m= PP_SVGPFA(T,Y,mask,ks,Zs,C0,d0,q_diag=True,link=tf_link)

#---------------------------------------------------

sess  = tf.Session()
sess.run(tf.global_variables_initializer()) # reset values to wrong
# declare loss
loss = -m.build_likelihood()
# separate variables
vars_e, vars_m, vars_h, vars_z= [], [], [], []
vars_e += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/inference')
vars_m += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/param')
vars_z += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ind_points')
vars_h += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='kernels')
# declare optimizers
opt_e = soi(loss, var_list=vars_e,  method='L-BFGS-B', options={'ftol': 1e-4})
opt_m = soi(loss, var_list=vars_m,  method='L-BFGS-B', options={'ftol': 1e-4})
opt_z = soi(loss, var_list=vars_z,  method='L-BFGS-B', options={'ftol': 1e-4})
opt_h = soi(loss, var_list=vars_h,  method='L-BFGS-B', options={'ftol': 1e-4})

init = tf.global_variables_initializer()
sess.run(init) # reset values to wrong
feed_dic = {Y:Ts_mat, mask:mask_mat}

#---------------------------------------------------

fig,axarr = plt.subplots(2,2,figsize=(10,8))

print('Optimized variables:')
for var in vars_e+vars_m+vars_h:
    print(var.name)  # Prints the name of the variable alongside its val

nit = 50
loss_array = np.zeros((nit,))
x = np.linspace(0,T,500).reshape(-1,1).astype(np_float_type)


# declare which optimization to perform
OPT = ['E','Z','M','H']
# Optimization is performed using L-BFGS-B, iterating over different subsets of variable
# - E: inference (as in classical EM)
# - M: updates for loading C, offest d (as in classical EM)
# - Z: update of inducing point locations
# - H: kernel hyperparameter optimization

print('Starting Optimization')
opt_e.minimize(sess, feed_dict=feed_dic)
if 'E' in OPT:
    print('1rst E_step')
    opt_m.minimize(sess, feed_dict=feed_dic)
if 'H' in OPT:
    print('1rst H_step')
    opt_h.minimize(sess, feed_dict=feed_dic)

for it in range( nit):
    if 'E' in OPT:
        print('E_step %d/%d ' % (it, nit))
        opt_e.minimize(sess, feed_dict=feed_dic)
    if 'Z' in OPT:
        print('Z_step %d/%d ' % (it, nit))
        opt_z.minimize(sess, feed_dict=feed_dic)
    if 'M' in OPT:
        print('M_step %d/%d ' % (it, nit))
        opt_m.minimize(sess, feed_dict=feed_dic)
    if 'H' in OPT:
        print('H_step %d/%d ' % (it, nit))
        opt_z.minimize(sess, feed_dict=feed_dic)

    loss_array[it]= float(sess.run(loss, feed_dic))

    Fs_mean,Fs_var = sess.run(m.build_predict_fs(x),  feed_dic)
    Zs = sess.run(m.Zs,  feed_dic)
    y_mean,y_var = sess.run(m.predict_log_rates(x),  feed_dic)
    ax = axarr[0,0]
    for d in range(D):
        for r in range(R):
            f,s =  Fs_mean[d,:,r],np.sqrt(Fs_var[d,:,r])
            ax.vlines(Zs[d],ymin=f.min(),ymax=f.max(),color=colors_d[d],alpha=.05)
            ax.plot(x,f,color=colors_d[d])
            ax.plot(x,fs[d](x),color=colors_d[d],linestyle='--')
            ax.fill_between(x.flatten(),f-s,y2=f+s,alpha=.3,facecolor=colors_d[d])
    ax.set_xlabel('time (s)')
    ax.set_title('true and inferred latents')
    ax = axarr[0,1]
    for o in range(O):
        for r in range(R):
            y = y_mean[:, o, r]
            ax.plot(log_rates[o](x).flatten(), y.flatten(), color=colors_o[o])
    ax.plot([y_mean.min(),y_mean.max()],[y_mean.min(),y_mean.max()],'k--',linewidth=3,alpha=.5)
    ax.set_xlabel('log rate (true)')
    ax.set_ylabel('log rate (predicted)')

    ax=axarr[1,0]
    ax.plot(loss_array[:it], linewidth=3, color='blue')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Variational Objective')

    ax=axarr[1,1]
    ax.imshow(y_mean[:, :, 0].T, interpolation='nearest',extent=[0,T,0,O],
                                origin='lower', aspect='auto', cmap=cm.gray)
    ax.set_xlabel('time')
    ax.set_ylabel('neuron index')


    fig.tight_layout()
    fig.savefig('%s_results.pdf' % model)
    plt.close()

#=================================


