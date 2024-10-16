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
from numpy.linalg import lstsq, norm
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
    np.random.seed(seed) # fix Random seed!
    # --- Parameter for DRLQC --- #
    tol = 1e-2
    # --- ----- --------#
    #noisedist = [noise_dist1]
    num_noise_list = [num_noise_samples]
    
    output_J_LQG_mean, output_J_WDRC_mean, output_J_DRCE_mean, output_J_DRLQC_mean =[], [], [], []
    output_J_LQG_std, output_J_WDRC_std, output_J_DRCE_std, output_J_DRLQC_std=[], [], [], []
    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = 0.2*(np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B= np.eye(10)
    C = Q = R = Qf = np.eye(10) 
    #----------------------------
    # You can change theta_v list and lambda_list ! but you also need to change lists at plot_params4_drlqc_nonzeromean.py to get proper plot
    
    if dist=='normal':
        theta_v_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # radius of noise ambiguity set
        theta_w_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        theta_v_list = [2.0, 4.0, 6.0] # radius of noise ambiguity set
        theta_w_list = [2.0, 4.0, 6.0] # radius of noise ambiguity set
        #theta_w_list = [6.0]
    else:
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # radius of noise ambiguity set
        theta_w_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        #theta_v_list = [1.0, 2.0, 3.0] # radius of noise ambiguity set
        #theta_w_list = [1.0, 2.0, 3.0] # radius of noise ambiguity set
    lambda_list = [3] # disturbance distribution penalty parameter # not used if use_lambda = False
    theta_w = 1.0 # will not be used if use_lambda = True
    #num_x0_samples = 15 #  N_x0 
    theta_x0 = 1.0 # radius of initial state ambiguity set
    use_lambda = False # If use_lambda=True, we will use lambda_list. If use_lambda=False, we will use theta_w_list
    use_optimal_lambda = False
    if use_lambda:
        dist_parameter_list = lambda_list
    else:
        dist_parameter_list = theta_w_list
    
    # Lambda list (from the given theta_w, WDRC and WDR-CE calcluates optimized lambda)
    if dist=="normal":
        WDRC_lambda_file = open('./inputs/io_nn/nonzero_wdrc_lambda.pkl', 'rb')
    elif dist=="quadratic":
        WDRC_lambda_file = open('./inputs/io_qq/nonzero_wdrc_lambda.pkl', 'rb')
    WDRC_lambda = pickle.load(WDRC_lambda_file)
    WDRC_lambda_file.close()
    if dist=="normal":
        DRCE_lambda_file = open('./inputs/io_nn/nonzero_drce_lambda.pkl', 'rb')
    elif dist=="quadratic":
        DRCE_lambda_file = open('./inputs/io_qq/nonzero_drce_lambda.pkl', 'rb')
    DRCE_lambda = pickle.load(DRCE_lambda_file)
    DRCE_lambda_file.close()
    
    print("WDRC_lambda")
    print(WDRC_lambda)
    # Uncomment Below 2 lines to save optimal lambda, using your own distributions.
    WDRC_lambda = np.zeros((len(theta_w_list),len(theta_v_list)))
    DRCE_lambda = np.zeros((len(theta_w_list),len(theta_v_list)))
    #-------Disturbance Distribution-------
    if dist == "normal":
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = 0.3*np.ones((nx, 1))
        Sigma_w= 0.8*np.eye(nx)
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = 0.2*np.ones((nx,1))
        x0_cov = 0.001*np.eye(nx)
    elif dist == "quadratic":
        #disturbance distribution parameters
        w_max = 1.5*np.ones(nx)
        w_min = -1.2*np.ones(nx)
        mu_w = (0.5*(w_max + w_min))[..., np.newaxis]
        Sigma_w = 3.0/20.0*np.diag((w_max - w_min)**2)
        #initial state distribution parameters
        x0_max = 0.21*np.ones(nx)
        x0_min = 0.19*np.ones(nx)
        x0_mean = (0.5*(x0_max + x0_min))[..., np.newaxis]
        x0_cov = 3.0/20.0 *np.diag((x0_max - x0_min)**2)
        
    #-------Noise distribution ---------#
    if noise_dist =="normal":
        v_max = None
        v_min = None
        M = 3.0*np.eye(ny) #observation noise covariance
        mu_v = 0.1*np.ones((ny, 1))
    elif noise_dist =="quadratic":
        v_min = -1.5*np.ones(ny)
        v_max = 2.5*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
    
    x0 = x0_mean    
    N=1000
    # -------Estimate the nominal distribution-------
    # Initialize estimates
    mu_w_hat = np.zeros((nx, 1))
    C_w = np.zeros((nx, nx))  # Scatter matrix for w
    mu_v_hat = np.zeros((ny, 1))
    C_v = np.zeros((ny, ny))  # Scatter matrix for v
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    # Counters for recursive estimation
    k_w = 0  # Number of samples for w
    k_v = 0  # Number of samples for v

    # Kalman Filter Initialization
    P = np.eye(nx)  # Initial estimate error covariance

    epsilon = 1e-8  # Small value to ensure positive definiteness

    for i in range(N):
        print(f"Sequence {i+1}/{N}")
        # Initialize state estimate and covariance
        x_hat = np.zeros((T + 1, nx, 1))
        x_hat[0] = x0  # Known initial state
        P = np.eye(nx)

        for t in range(T):
            # Sample control input u_t from zero-mean Gaussian distribution
            u_t = np.random.multivariate_normal(np.zeros(nu), np.eye(nu)).reshape(nu, 1)

            # Sample true process noise and measurement noise
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)

            # State update (true system)
            if t == 0:
                x_true = x0
            else:
                x_true = x_next_true

            x_next_true = A @ x_true + B @ u_t + true_w

            # Measurement
            y_t = C @ x_true + true_v

            # Predict Step
            x_pred = A @ x_hat[t] + B @ u_t + mu_w_hat
            P_pred = A @ P @ A.T + Sigma_w_hat + epsilon * np.eye(nx)

            # Measurement Update Step
            y_pred = C @ x_pred + mu_v_hat
            S = C @ P_pred @ C.T + Sigma_v_hat + epsilon * np.eye(ny)

            # Check for singularity and adjust if necessary
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
                print("Warning: S matrix is singular; using pseudo-inverse.")

            K = P_pred @ C.T @ S_inv

            # Update estimate
            x_hat[t + 1] = x_pred + K @ (y_t - y_pred)
            P = (np.eye(nx) - K @ C) @ P_pred

            # Compute residuals
            w_t_hat = x_hat[t + 1] - A @ x_hat[t] - B @ u_t - mu_w_hat
            v_t_hat = y_t - C @ x_hat[t] - mu_v_hat

            # Update estimates recursively using Welford's algorithm
            # Update mu_w_hat and C_w
            k_w += 1
            delta_w = w_t_hat - mu_w_hat
            mu_w_hat += delta_w / k_w
            delta_w2 = w_t_hat - mu_w_hat
            C_w += delta_w @ delta_w2.T

            # Update mu_v_hat and C_v
            k_v += 1
            delta_v = v_t_hat - mu_v_hat
            mu_v_hat += delta_v / k_v
            delta_v2 = v_t_hat - mu_v_hat
            C_v += delta_v @ delta_v2.T

            # Update covariance estimates
            if k_w > 1:
                Sigma_w_hat = C_w / (k_w - 1) + epsilon * np.eye(nx)
            else:
                Sigma_w_hat = np.eye(nx)

            if k_v > 1:
                Sigma_v_hat = C_v / (k_v - 1) + epsilon * np.eye(ny)
            else:
                Sigma_v_hat = np.eye(ny)

        # Optionally, print estimates after each sequence
        print(f"After sequence {i+1}:")
        print(f"Estimated mu_w_hat:\n{mu_w_hat}\n")
        print(f"Estimated Sigma_w_hat:\n{Sigma_w_hat}\n")
        print(f"Estimated mu_v_hat:\n{mu_v_hat}\n")
        print(f"Estimated Sigma_v_hat:\n{Sigma_v_hat}\n")

    # --- Quantify Estimation Errors ---

    # Mean estimation errors (Euclidean norms)
    error_mu_w = np.linalg.norm(mu_w_hat - mu_w)
    error_mu_v = np.linalg.norm(mu_v_hat - mu_v)

    # Covariance estimation errors (Frobenius norms)
    error_Sigma_w = np.linalg.norm(Sigma_w_hat - Sigma_w, 'fro')
    error_M = np.linalg.norm(Sigma_v_hat - M, 'fro')
    
    M_hat = Sigma_v_hat

    # --- Results ---

    print("Estimated mu_w:")
    print(mu_w_hat)
    print("\nTrue mu_w:")
    print(mu_w)
    print("\nEstimation Error (mu_w): {:.6f}".format(error_mu_w))

    print("\nEstimated Sigma_w:")
    print(Sigma_w_hat)
    print("\nTrue Sigma_w:")
    print(Sigma_w)
    print("\nEstimation Error (Sigma_w): {:.6f}".format(error_Sigma_w))

    print("\nEstimated mu_v:")
    print(mu_v_hat)
    print("\nTrue mu_v:")
    print(mu_v)
    print("\nEstimation Error (mu_v): {:.6f}".format(error_mu_v))

    print("\nEstimated M:")
    print(M_hat)
    print("\nTrue M:")
    print(M)
    print("\nEstimation Error (M): {:.6f}".format(error_M))
    
    #exit()
    
    # ----- Construct Batch matrix for DRLQC-------------------
    W_hat = np.zeros((nx, nx, T+1))
    V_hat = np.zeros((ny, ny, T+1))
    for i in range(T):
        W_hat[:,:,i] = Sigma_w_hat
        V_hat[:,:,i] = M_hat
    # ----------------------------
    mu_w_hat = np.tile(mu_w_hat, (T,1,1) )
    mu_v_hat = np.tile(mu_v_hat, (T+1,1,1) )
    Sigma_w_hat = np.tile(Sigma_w_hat, (T,1,1))
    M_hat = np.tile(M_hat, (T+1,1,1))
    x0_mean_hat = x0_mean # Assume known initial state for this experiment
    x0_cov_hat = x0_cov
    
     # Create paths for saving individual results
    temp_results_path = "./temp_results/"
    if not os.path.exists(temp_results_path):
        os.makedirs(temp_results_path)
        
    def perform_simulation(lambda_, noise_dist, dist_parameter, theta, idx_w, idx_v):
        for num_noise in num_noise_list:
            np.random.seed(seed) # fix Random seed!
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
                path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/ioreal/".format(dist, noise_dist)
            else:
                path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/ioreal/".format(dist, noise_dist)
                
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
            drlqc = DRLQC(theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_w_hat, W_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat, V_hat, x0_mean_hat, x0_cov_hat, tol)
            if use_optimal_lambda == True:
                lambda_ = WDRC_lambda[idx_w][idx_v]
            #print(lambda_)
            wdrc = WDRC(lambda_, theta_w, T, dist, noise_dist, system_data, mu_w_hat, Sigma_w_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat, use_lambda, use_optimal_lambda)
            if use_optimal_lambda == True:
                lambda_ = DRCE_lambda[idx_w][idx_v]
            drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_w_hat, Sigma_w_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat,  M_hat, x0_mean_hat, x0_cov_hat, use_lambda, use_optimal_lambda)
            lqg = LQG(T, dist, noise_dist, system_data, mu_w_hat, Sigma_w_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat, M_hat , x0_mean_hat, x0_cov_hat)
        
            drlqc.solve_sdp()
            drlqc.backward()
            wdrc.backward()
            drce.backward()
            lqg.backward()
            
            # Save the optimzed lambda : Uncomment below two lines if you want to save optimal lambda for your own distributions
            WDRC_lambda[idx_w][idx_v] = wdrc.lambda_
            DRCE_lambda[idx_w][idx_v] = drce.lambda_
            # Save individual lambda results for debugging and validation
            wdrc_lambda_filename = os.path.join(temp_results_path, f'wdrc_lambda_{idx_w}_{idx_v}.pkl')
            drce_lambda_filename = os.path.join(temp_results_path, f'drce_lambda_{idx_w}_{idx_v}.pkl')
            
            save_pickle_data(wdrc_lambda_filename, wdrc.lambda_)
            save_pickle_data(drce_lambda_filename, drce.lambda_)
            print(f"Saved WDRC_lambda[{idx_w}][{idx_v}] and DRCE_lambda[{idx_w}][{idx_v}]")

            print("WDRC_lambda[{}][{}] :{} ".format(idx_w, idx_v, wdrc.lambda_)  )
            print('---------------------')
            np.random.seed(seed) # fix Random seed!
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
            np.random.seed(seed) # fix Random seed!
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
            # Save cost data #
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
            
            #Save all raw data
            if use_lambda:
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/ioreal/raw/".format(dist, noise_dist)
            else:
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/ioreal/raw/".format(dist, noise_dist)
                
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
             
            
            #Summarize and plot the results
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
        path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/ioreal/".format(dist, noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/ioreal/".format(dist, noise_dist)
        save_data(path + 'nonzero_wdrc_lambda.pkl',WDRC_lambda)
        save_data(path + 'nonzero_drce_lambda.pkl',DRCE_lambda)
        
    
            
    print("Params data generation Completed !")
    print("Please make sure your lambda_list(or theta_w_list) and theta_v_list plot_params4_usingio_real.py is as desired")
    if use_lambda:
        print("Now use : python plot_params4_usingio_real.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        print("Now use : python plot_params4_usingio_real.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="quadratic", type=str) #disurbance distribution (normal or quadratic)
    parser.add_argument('--noise_dist', required=False, default="quadratic", type=str) #noise distribution (normal or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int) #number of simulation runs
    parser.add_argument('--num_samples', required=False, default=15, type=int) #number of disturbance samples
    parser.add_argument('--num_noise_samples', required=False, default=15, type=int) #number of noise samples
    parser.add_argument('--horizon', required=False, default=20, type=int) #horizon length
    
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_samples, args.num_noise_samples, args.horizon)
