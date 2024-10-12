#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file generates data for Nonzeromean U-Quadratic distributions
# 4 method implemented (LQG, WDRC, DRLQC, WDR-CE)

import numpy as np
import argparse
from controllers.LQG import LQG
from controllers.WDRC import WDRC
from controllers.DRCE import DRCE
from controllers.DRLQC import DRLQC
from joblib import Parallel, delayed

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


def save_pickle_data(path, data):
    """Save data using pickle to the specified path."""
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_pickle_data(path):
    """Load data using pickle from the specified path."""
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)
    
def save_data(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()

def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T):
    
    lambda_ = 10
    seed = 2024 # Random seed
    # --- Parameter for DRLQC --- #
    tol = 1e-2
    # --- ----- --------#
    #oisedist = [noise_dist1]
    num_noise_list = [num_noise_samples]
    # You can change theta_v list and lambda_list ! but you also need to change lists at plot_params4_drlqc_nonzeromean.py to get proper plot
    
    
    if dist=='normal':
        theta_v_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        theta_w_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
    else:
        theta_v_list = [0.1] # radius of noise ambiguity set
        theta_w_list = [0.01, 0.02] # radius of noise ambiguity set
        #theta_w_list = [0.1]
    lambda_list = [100000] # disturbance distribution penalty parameter
    num_x0_samples = 5 #  N_x0 
    theta_x0 = 1.0 # radius of initial state ambiguity set
    
    use_lambda = False # If use_lambda=True, we will use lambda_list. If use_lambda=False, we will use theta_w_list
    use_optimal_lambda = False
    if use_lambda:
        dist_parameter_list = lambda_list
    else:
        dist_parameter_list = theta_w_list
    
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean, output_J_DRLQC_mean =[], [], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std, output_J_DRLQC_std=[], [], [], []
    #-------Initialization-------
    nx = 4 #state dimension
    nu = 2 #control input dimension
    ny = 2 #output dimension
    # sideslip angle
    # roll rate
    # yaw rate
    # roll angle 
    A = np.array([
    [0.9801, 0.0003, -0.0980, 0.0038],
    [-0.3868, 0.9071, 0.0471, -0.0008],
    [0.1591, -0.0015, 0.9691, 0.0003],
    [-0.0198, 0.0958, 0.0021, 1.000]
    ])

    B = np.array([
    [-0.0001, 0.0058],
    [0.0296, 0.0153],
    [0.0012, -0.0908],
    [0.0015, 0.0008]
    ])

    # C matrix 
    C = np.array([
    [1,0,0,0],
    [0,0,0,1]
    ])
    
    # Check Controllability Matrix
    #ctrb_matrix = control.ctrb(A, B)
    #print(np.linalg.matrix_rank(ctrb_matrix))

    # Check Observability Matrix
    #obsv_matrix = control.obsv(A, C)
    #print(np.linalg.matrix_rank(obsv_matrix))
    #exit()

    Qf = Q = np.array([
    [1, 0, 0, 0],
    [0, 10, 0, 0],
    [0, 0, 30, 0],
    [0, 0, 0, 30]
    ])

    #u0: aileron deflection
    #u1: rudder deflection
    R = np.array([
    [0.03, 0],
    [0, 3.16]
    ])
    #-------Disturbance Distribution-------
    disturbance_scale = np.array([
    0.01,#sideslip angle
    0.02,#roll rate
    0.02,#yaw rate
    0.02 #roll angle
    ])
    disturbance_scale = disturbance_scale.reshape((4, 1))
    measurement_noise_scale = np.array([
    0.005,
    0.01
    ])
    measurement_noise_scale=measurement_noise_scale.reshape((2,1))
    
    #-------Disturbance Distribution-------
    if dist == "normal":
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = 0.3*np.ones((nx, 1))
        Sigma_w= 0.6*np.eye(nx)
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = 0.5*np.ones((nx,1))
        x0_cov = 0.001*np.eye(nx)
    elif dist == "quadratic":
        #disturbance distribution parameters
        w_max = 1*disturbance_scale
        w_min = -1*disturbance_scale
        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
        #initial state distribution parameters
        x_0 = np.zeros((nx,1))
        x_0[3] = 0.1 # roll angle ~6,7 degree 
        x0_max = x_0+0.1*disturbance_scale
        x0_min = x_0-0.1*disturbance_scale
        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
        
    #-------Noise distribution ---------#
    if noise_dist =="normal":
        v_max = None
        v_min = None
        M = 3.5*np.eye(ny) #observation noise covariance
        mu_v = 0.1*np.ones((ny, 1))
    elif noise_dist =="quadratic":
        v_min = -0.1*measurement_noise_scale
        v_max = 0.1*measurement_noise_scale
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance

    #-------Estimate the nominal distribution-------
    # Nominal initial state distribution
    x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
    # Nominal Disturbance distribution
    mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
    # Nominal Noise distribution
    v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise_samples, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)

    M_hat = M_hat + 1e-8*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
    Sigma_hat = Sigma_hat + 1e-8*np.eye(nx)
    x0_cov_hat = x0_cov_hat + 1e-8*np.eye(nx) 
    
    # ----- Construct Batch matrix for DRLQC-------------------
    W_hat = np.zeros((nx, nx, T+1))
    V_hat = np.zeros((ny, ny, T+1))
    for i in range(T):
        W_hat[:,:,i] = Sigma_hat[i]
        V_hat[:,:,i] = M_hat[i]
    # ----------------------------
    #----------------------------
    
    # Lambda list (from the given theta_w, WDRC and WDR-CE calcluates optimized lambda)
    WDRC_lambda_file = open('./inputs/nonzero_qq/nonzero_wdrc_lambda.pkl', 'rb')
    WDRC_lambda = pickle.load(WDRC_lambda_file)
    WDRC_lambda_file.close()
    DRCE_lambda_file = open('./inputs/nonzero_qq/nonzero_drce_lambda.pkl', 'rb')
    DRCE_lambda = pickle.load(DRCE_lambda_file)
    DRCE_lambda_file.close()
    
    # Uncomment Below 2 lines to save optimal lambda, using your own distributions.
    WDRC_lambda = np.zeros((len(theta_w_list),len(theta_v_list)))
    DRCE_lambda = np.zeros((len(theta_w_list),len(theta_v_list)))
    
    # Create paths for saving individual results
    temp_results_path = "./temp_results/"
    if not os.path.exists(temp_results_path):
        os.makedirs(temp_results_path)
    def perform_simulation(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v):
        for num_noise in num_noise_list:
            np.random.seed(seed) # fix Random seed!
            #theta_w = 1.0 # Will not be used if use_lambda = True, placeholder
            print("--------------------------------------------")
            print("number of noise sample : ", num_noise)
            print("number of disturbance sample : ", num_samples)
            if use_lambda:
                lambda_ = dist_parameter
                print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ lambda: ", lambda_, "/ theta_v : ", theta)
            else:
                theta_w = dist_parameter
                print("disturbance : ", dist, "/ noise : ", noise_dist, "/ num_noise : ", num_noise, "/ theta_w: ", theta_w, "/ theta_v : ", theta)

            if use_lambda:
                path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/747/".format(dist, noise_dist)
            else:
                path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/747/".format(dist, noise_dist)
                
            if not os.path.exists(path):
                os.makedirs(path)
        
            
            
            
            #-------Create a random system-------
            system_data = (A, B, C, Q, Qf, R, M)
            
            #-------Perform n independent simulations and summarize the results-------
            output_lqg_list = []
            output_wdrc_list = []
            output_drce_list = []
            output_drlqc_list = []
            
            #-----Initialize controllers-----
            drlqc = DRLQC(theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, W_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, V_hat, x0_mean_hat[0], x0_cov_hat[0], tol)
            if use_optimal_lambda == True:
                lambda_ = WDRC_lambda[idx_w][idx_v]
            #print(lambda_)
            wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
            if use_optimal_lambda == True:
                lambda_ = DRCE_lambda[idx_w][idx_v]
            drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat,  M_hat, x0_mean_hat[0], x0_cov_hat[0], use_lambda, use_optimal_lambda)
            lqg = LQG(T, dist, noise_dist, system_data, mu_hat, Sigma_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, v_mean_hat, M_hat , x0_mean_hat[0], x0_cov_hat[0])
        
            drlqc.solve_sdp()
            drlqc.backward()
            wdrc.backward()
            drce.backward()
            lqg.backward()
            
            # Save the optimzed lambda : Uncomment below two lines if you want to save optimal lambda for your own distributions
            WDRC_lambda[idx_w][idx_v] = wdrc.lambda_
            DRCE_lambda[idx_w][idx_v] = drce.lambda_
            print('---------------------')
            
            #----------------------------
            print("Running DRCE Forward step ...")
            for i in range(num_sim):
                
                #Perform state estimation and apply the controller
                output_drce = drce.forward()
                output_drce_list.append(output_drce)
                if i%50==0:
                    print("Simulation #",i, ' | cost (DRCE):', output_drce['cost'][0], 'time (DRCE):', output_drce['comp_time'])
            
            J_DRCE_list = []
            for out in output_drce_list:
                J_DRCE_list.append(out['cost'])
            J_DRCE_mean= np.mean(J_DRCE_list, axis=0)
            J_DRCE_std = np.std(J_DRCE_list, axis=0)
            output_J_DRCE_mean.append(J_DRCE_mean[0])
            output_J_DRCE_std.append(J_DRCE_std[0])
            print(" Average cost (DRCE) : ", J_DRCE_mean[0])
            print(" std (DRCE) : ", J_DRCE_std[0])
            
            #----------------------------
            print("Running DRLQC Forward step ...")
            for i in range(num_sim):
                
                #Perform state estimation and apply the controller
                output_drlqc = drlqc.forward()
                output_drlqc_list.append(output_drlqc)
                if i%50==0:
                    print("Simulation #",i, ' | cost (DRLQC):', output_drlqc['cost'][0], 'time (DRLQC):', output_drlqc['comp_time'])
            
            J_DRLQC_list = []
            for out in output_drlqc_list:
                J_DRLQC_list.append(out['cost'])
            J_DRLQC_mean= np.mean(J_DRLQC_list, axis=0)
            J_DRLQC_std = np.std(J_DRLQC_list, axis=0)
            output_J_DRLQC_mean.append(J_DRLQC_mean[0])
            output_J_DRLQC_std.append(J_DRLQC_std[0])
            print(" Average cost (DRLQC) : ", J_DRLQC_mean[0])
            print(" std (DRLQC) : ", J_DRLQC_std[0])
            
            #----------------------------             
            np.random.seed(seed) # fix Random seed!
            print("Running WDRC Forward step ...")  
            for i in range(num_sim):
        
                #Perform state estimation and apply the controller
                output_wdrc = wdrc.forward()
                output_wdrc_list.append(output_wdrc)
                if i%50==0:
                    print("Simulation #",i, ' | cost (WDRC):', output_wdrc['cost'][0], 'time (WDRC):', output_wdrc['comp_time'])
            
            J_WDRC_list = []
            for out in output_wdrc_list:
                J_WDRC_list.append(out['cost'])
            J_WDRC_mean= np.mean(J_WDRC_list, axis=0)
            J_WDRC_std = np.std(J_WDRC_list, axis=0)
            output_J_WDRC_mean.append(J_WDRC_mean[0])
            output_J_WDRC_std.append(J_WDRC_std[0])
            print(" Average cost (WDRC) : ", J_WDRC_mean[0])
            print(" std (WDRC) : ", J_WDRC_std[0])
            #----------------------------
            np.random.seed(seed) # fix Random seed!
            print("Running LQG Forward step ...")
            for i in range(num_sim):
                output_lqg = lqg.forward()
                output_lqg_list.append(output_lqg)
        
                if i%50==0:
                    print("Simulation #",i, ' | cost (LQG):', output_lqg['cost'][0], 'time (LQG):', output_lqg['comp_time'])
                
            J_LQG_list = []
            for out in output_lqg_list:
                J_LQG_list.append(out['cost'])
            J_LQG_mean= np.mean(J_LQG_list, axis=0)
            J_LQG_std = np.std(J_LQG_list, axis=0)
            output_J_LQG_mean.append(J_LQG_mean[0])
            output_J_LQG_std.append(J_LQG_std[0])
            print(" Average cost (LQG) : ", J_LQG_mean[0])
            print(" std (LQG) : ", J_LQG_std[0])
            
            #-----------------------------------------
            # Save data #
            theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
            theta_w_ = f"_{str(theta_w).replace('.', '_')}" # change 1.0 to 1_0 for file name
            if use_lambda:
                save_data(path + 'drce_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                save_data(path + 'drlqc_' + str(lambda_) + 'and' + theta_v_+ '.pkl', J_DRLQC_mean)
                save_data(path + 'wdrc_' + str(lambda_) + '.pkl', J_WDRC_mean)
            else:
                save_data(path + 'drlqc' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRLQC_mean)
                save_data(path + 'drce' + theta_w_ + 'and' + theta_v_+ '.pkl', J_DRCE_mean)
                save_data(path + 'wdrc' + theta_w_ + '.pkl', J_WDRC_mean)
                
            save_data(path + 'lqg.pkl', J_LQG_mean)
    
            #save_data(path + 'nonzero_wdrc_lambda.pkl',WDRC_lambda)
            #save_data(path + 'nonzero_drce_lambda.pkl',DRCE_lambda)
            #Summarize and plot the results
            #Save all raw data
            if use_lambda:
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/nonzeropp/raw/".format(dist, noise_dist)
            else:
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/nonzeropp/raw/".format(dist, noise_dist)
                
            if not os.path.exists(rawpath):
                os.makedirs(rawpath)
                
            if use_lambda:
                save_data(rawpath + 'drce_' + str(lambda_) + 'and' + theta_v_+ '.pkl', output_drce_list)
                save_data(rawpath + 'drlqc_' + str(lambda_) + 'and' + theta_v_+ '.pkl', output_drlqc_list) # lambda not used
                save_data(rawpath + 'wdrc_' + str(lambda_) + '.pkl', output_wdrc_list)
            else:
                save_data(rawpath + 'drlqc' + theta_w_ + 'and' + theta_v_+ '.pkl', output_drlqc_list)
                save_data(rawpath + 'drce' + theta_w_ + 'and' + theta_v_+ '.pkl', output_drce_list)
                save_data(rawpath + 'wdrc' + theta_w_ + '.pkl', output_wdrc_list)
                
            save_data(rawpath + 'lqg.pkl', output_lqg_list)
            print('\n-------Summary-------')
            print("dist : ", dist,"/ noise dist : ", noise_dist, "/ num_samples : ", num_samples, "/ num_noise_samples : ", num_noise, "/seed : ", seed)
    
    combinations = [(dist_parameter, theta, idx_w, idx_v) for idx_w, dist_parameter in enumerate(dist_parameter_list) for idx_v, theta in enumerate(theta_v_list)]
    
    results = Parallel(n_jobs=-1)(
                delayed(perform_simulation)(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v)
                for dist_parameter, theta, idx_w, idx_v in combinations
            )     
    
    for idx_w in range(len(theta_w_list)):
        for idx_v in range(len(theta_v_list)):
            wdrc_lambda_filename = os.path.join(temp_results_path, f'wdrc_lambda_{idx_w}_{idx_v}.pkl')
            drce_lambda_filename = os.path.join(temp_results_path, f'drce_lambda_{idx_w}_{idx_v}.pkl')

            # Load individual results and update the final arrays
            if os.path.exists(wdrc_lambda_filename):
                WDRC_lambda[idx_w][idx_v] = load_pickle_data(wdrc_lambda_filename)
            if os.path.exists(drce_lambda_filename):
                DRCE_lambda[idx_w][idx_v] = load_pickle_data(drce_lambda_filename)
                
    if use_lambda:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/747/".format(dist, noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/747/".format(dist, noise_dist)
        save_data(path + 'nonzero_wdrc_lambda.pkl',WDRC_lambda)
        save_data(path + 'nonzero_drce_lambda.pkl',DRCE_lambda)
            
    print("Params data generation Completed !")
    print("Please make sure your lambda_list(or theta_w_list) and theta_v_list in plot_params4_drlqc_nonzeromean.py is as desired")
    if use_lambda:
        print("Now use : python plot_params4_drlqc_nonzeromean_p.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        print("Now use : python plot_params4_drlqc_nonzeromean_p.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="quadratic", type=str) #disurbance distribution (normal or quadratic)
    parser.add_argument('--noise_dist', required=False, default="quadratic", type=str) #noise distribution (normal or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=5, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=5, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
