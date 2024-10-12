#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file generates data for Gaussian(Normal) and U-Quadratic distributions
# 3 method implemented (LQG, WDRC, WDR-CE)
# DRLQC not implemented : Frank-Wolfe algorithm shows unstable behavior in this system

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE

import os
import pickle

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b-a)*np.random.rand(N,n)
    return x.T

def normal(mu, Sigma, N=1):
    x = np.random.multivariate_normal(mu[:,0], Sigma, size=N).T
    return x
def quad_inverse(x, b, a):
    row = x.shape[0]
    col = x.shape[1]
    for i in range(row):
        for j in range(col):
            beta = (a[j]+b[j])/2.0
            alpha = 12.0/((b[j]-a[j])**3)
            tmp = 3*x[i][j]/alpha - (beta - a[j])**3
            if 0<=tmp:
                x[i][j] = beta + ( tmp)**(1./3.)
            else:
                x[i][j] = beta -(-tmp)**(1./3.)
    return x

# quadratic U-shape distrubituon in [wmin , wmax]
def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    #print("wmax : " , wmax)
    x = quad_inverse(x, wmax, wmin)
    return x.T

def multimodal(mu, Sigma, N=1):
    modes = 2
    n = mu[0].shape[0]
    x = np.zeros((n,N,modes))
    for i in range(modes):
        w = np.random.normal(size=(N,n))
        if (Sigma[i] == 0).all():
            x[:,:,i] = mu[i]
        else:
            x[:,:,i] = mu[i] + np.linalg.cholesky(Sigma[i]) @ w.T

    #w = np.random.choice([0, 1], size=(n,N))
    w = 0.5
    y = x[:,:,0]*w + x[:,:,1]*(1-w)
    return y

def gen_sample_dist(dist, T, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

def gen_sample_dist_inf(dist, N_sample, mu_w=None, Sigma_w=None, w_max=None, w_min=None):
    if dist=="normal":
        w = normal(mu_w, Sigma_w, N=N_sample)
    elif dist=="uniform":
        w = uniform(w_max, w_min, N=N_sample)
    elif dist=="multimodal":
        w = multimodal(mu_w, Sigma_w, N=N_sample)
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)
        
    mean_ = np.average(w, axis = 1)[...,np.newaxis]
    var_ = np.cov(w)
    return mean_, var_


def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist1, num_sim, num_samples, num_noise_samples, T):
    noise_plot_results = True
    seed = 2024 # Random seed !  any value
    if noise_plot_results:
        num_noise_list = [5, 10, 15, 20, 25, 30, 35, 40]
    else:
        num_noise_list = [num_noise_samples]
    num_x0_samples = 15 # num x0 samples 
    # for the noise_plot_results!!
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean=[], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std=[], [], []
    #-------Initialization-------
    nx = 21
    nu = 11
    ny = 10
    A = np.array([[-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,  0,	0,	0],
                    [0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	-1,	0,	0,  0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,  0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	0,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	0,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	0,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	0],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	-1],
                    [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1]
                    ])
    B = np.array([[1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	1,	0,  0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1]])
    C = np.zeros((10,21))
    C[0][1]=C[1][3]=C[2][5]=C[3][7]=C[4][9]=C[5][11]=C[6][13]=C[7][15]=C[8][17]=C[9][19]= 1
    Q = Qf = np.eye(21)
    R = np.eye(11) 
    #----------------------------
    # change True to False if you don't want to use given lambda
    use_lambda = True
    use_optimal_lambda = True
    lambda_ = 20 # will not be used if the parameter "use_lambda = False"
    noisedist = [noise_dist1]
    if dist == "normal":
        theta_w_list = [0.5]
        theta_v_list = [5.0]
        theta_x0 = 0.5 # radius of initial state ambiguity set
    elif dist == "quadratic":
        theta_w_list = [1.0]
        theta_v_list = [5.0]
        theta_x0 = 0.5
    else:
        theta_w_list = [0.5]
        theta_v_list = [5.0]
        theta_x0 = 0.5
    
    
    # Save lambda list
    WDRC_lambda, DRCE_lambda = [],[]
    if use_lambda == True and dist=="normal":
        WDRC_lambda = np.array([26.41121764, 26.16612111, 26.21934116,26.21934116, 26.36159012, 26.3621963, 26.42347884, 26.43546194, 26.42452678 ])
        DRCE_lambda = np.array([34.91749428, 40.98912609, 39.05410236, 38.13329201, 38.13329201, 46.68884815, 38.21829064, 45.01168795, 39.02915113])
    if use_lambda == True and dist=="quadratic":
        WDRC_lambda = np.array([18.73975602, 18.67355442, 18.65363162, 18.68274912, 18.67348403, 18.69286136, 18.69810686, 18.7043255])
        DRCE_lambda = np.array([22.9903849, 23.78663288, 23.88587422,23.72079564, 24.0604179, 24.02581889, 24.07412486, 24.08671891])
        
    for noise_dist in noisedist:
        for theta_w in theta_w_list:
            for theta in theta_v_list:
                for idx, num_noise in enumerate(num_noise_list):
                    
                    print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w : ", theta_w, "/ theta_v : ", theta)
                    np.random.seed(seed) # fix Random seed!
                    print("--------------------------------------------")
                    print("number of noise sample : ", num_noise)
                    print("number of disturbance sample : ", num_samples)
                    
                    path = "./results/{}_{}/finite/multiple/".format(dist, noise_dist)    
                    if not os.path.exists(path):
                        os.makedirs(path)
                
                    #-------Disturbance Distribution-------
                    if dist == "normal":
                        #disturbance distribution parameters
                        w_max = None
                        w_min = None
                        mu_w = 1.0*np.ones((nx, 1))
                        Sigma_w= 0.1*np.eye(nx)
                        #initial state distribution parameters
                        x0_max = None
                        x0_min = None
                        x0_mean = 0.1*np.ones((nx,1))
                        x0_cov = 0.1*np.eye(nx)
                    elif dist == "quadratic":
                        #disturbance distribution parameters
                        w_max = 0.8*np.ones(nx)
                        w_min = -0.4*np.ones(nx)
                        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
                        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
                        #initial state distribution parameters
                        x0_max = 0.5*np.ones(nx)
                        x0_min = 0.0*np.ones(nx)
                        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
                        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
                        
                    #-------Noise distribution ---------#
                    if noise_dist =="normal":
                        v_max = None
                        v_min = None
                        M = 1.0*np.eye(ny) #observation noise covariance
                        mu_v = 0.5*np.ones((ny, 1))
                    elif noise_dist =="quadratic":
                        v_min = -1.0*np.ones(ny)
                        v_max = 1.5*np.ones(ny)
                        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
                        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
                        
                        
                    #-------Estimate the nominal distribution-------
                    # Nominal initial state distribution
                    x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
                    # Nominal Disturbance distribution
                    mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
                    # Nominal Noise distribution
                    v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)
                    
                    #print(x0_mean_hat)
                    M_hat = M_hat + 1e-6*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
                    
                    #-------Create a random system-------
                    system_data = (A, B, C, Q, Qf, R, M)
                    
                    #-------Perform n  independent simulations and summarize the results-------
                    output_lqg_list = []
                    output_wdrc_list = []
                    output_drce_list = []
                    
                    #Initialize controllers
                    if use_lambda==True:
                        lambda_ = WDRC_lambda[idx]
                    wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    if use_lambda==True:
                        lambda_ = DRCE_lambda[idx]
                    drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
                    lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat , x0_mean_hat[0], x0_cov_hat[0])

                    # Save Lambda for later use!!
                    if use_lambda == False:
                        WDRC_lambda.append( wdrc.lambda_ )
                        DRCE_lambda.append( drce.lambda_)
                    
                    drce.backward()
                    wdrc.backward()
                    lqg.backward()
                        
                    print('---------------------')
                    
                    #----------------------------
                    print("Running DRCE Forward step ...")
                    for i in range(num_sim):
                        
                        #Perform state estimation and apply the controller
                        output_drce = drce.forward()
                        output_drce_list.append(output_drce)
                    
                        print('cost (DRCE):', output_drce['cost'][0], 'time (DRCE):', output_drce['comp_time'])
                    
                    #----------------------------             
                    np.random.seed(seed) # fix Random seed!
                    print("Running WDRC Forward step ...")  
                    for i in range(num_sim):
                
                        #Perform state estimation and apply the controller
                        output_wdrc = wdrc.forward()
                        output_wdrc_list.append(output_wdrc)
                        print('cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
                    
                    #----------------------------
                    np.random.seed(seed) # fix Random seed!
                    print("Running LQG Forward step ...")
                    for i in range(num_sim):
                        
                        #Perform state estimation and apply the controller
                        output_lqg = lqg.forward()
                        output_lqg_list.append(output_lqg)
                        print('cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
                    
                    
                
                    if noise_plot_results:
                        J_LQG_list, J_WDRC_list, J_DRCE_list= [], [], []
                        
                        #lqg-----------------------
                        for out in output_lqg_list:
                            J_LQG_list.append(out['cost'])
                            
                        J_LQG_mean= np.mean(J_LQG_list, axis=0)
                        J_LQG_std = np.std(J_LQG_list, axis=0)
                        output_J_LQG_mean.append(J_LQG_mean[0])
                        output_J_LQG_std.append(J_LQG_std[0])
                        print(" Average cost (LQG) : ", J_LQG_mean[0])
                        print(" std (LQG) : ", J_LQG_std[0])
                        
                        #wdrc-----------------------
                        for out in output_wdrc_list:
                            J_WDRC_list.append(out['cost'])
                            
                        J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
                        J_WDRC_std = np.std(J_WDRC_list, axis=0)
                        output_J_WDRC_mean.append(J_WDRC_mean[0])
                        output_J_WDRC_std.append(J_WDRC_std[0])
                        print(" Average cost (WDRC) : ", J_WDRC_mean[0])
                        print(" std (WDRC) : ", J_WDRC_std[0])
                        
                        #drce---------------------
                        for out in output_drce_list:
                            J_DRCE_list.append(out['cost'])
                            
                        J_DRCE_mean= np.mean(J_DRCE_list, axis=0)
                        J_DRCE_std = np.std(J_DRCE_list, axis=0)
                        output_J_DRCE_mean.append(J_DRCE_mean[0])
                        output_J_DRCE_std.append(J_DRCE_std[0])
                        print(" Average cost (DRCE) : ", J_DRCE_mean[0])
                        print(" std (DRCE) : ", J_DRCE_std[0])
                        
                        print("num_noise_sample : ", num_noise, " / finished with dist : ", dist, "/ noise_dist : ", noise_dist, "/ seed : ", seed)
                    else:
                        path = "./results/{}_{}/finite/multiple/".format(dist, noise_dist)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        save_data(path + 'drce.pkl', output_drce_list)
                        save_data(path + 'wdrc.pkl', output_wdrc_list)
                        save_data(path + 'lqg.pkl', output_lqg_list)
                
                        print('\n-------Summary-------')
                        print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
                        
                        
                # after running noise_samples lists!
                if noise_plot_results:
                    
                    path = "./results/{}_{}/finite/multiple/num_noise_plot/".format(dist, noise_dist)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    save_data(path + 'drce_mean.pkl', output_J_DRCE_mean)
                    save_data(path + 'drce_std.pkl', output_J_DRCE_std)  
                    save_data(path + 'lqg_mean.pkl', output_J_LQG_mean)
                    save_data(path + 'lqg_std.pkl', output_J_LQG_std) 
                    save_data(path + 'wdrc_mean.pkl', output_J_WDRC_mean)
                    save_data(path + 'wdrc_std.pkl', output_J_WDRC_std) 
                    
                    save_data(path + 'noiseplot_wdrc_lambda.pkl',WDRC_lambda)
                    save_data(path + 'noiseplot_drce_lambda.pkl',DRCE_lambda)
                    
                    #Summarize and plot the results
                    print('\n-------Summary-------')
                    print("dist : ", dist, "noise_dist : ", noise_dist, "/ num_disturbance_samples : ", num_samples, "/ theta_v : ", theta, " / noise sample effect PLOT / Seed : ",seed)
                    
                    # reset
                    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean=[], [], []
                    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std=[], [], []
                    
    print("Data generation Completed!!")
    
    if noise_plot_results:
        print("For noise sample size effect plot Use : python plot_J.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
