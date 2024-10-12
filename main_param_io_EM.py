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
# Functions used in the EM algorithm

def kalman_smoother(u, y, A, B, C, mu_w, Sigma_w, mu_v, Sigma_v, x0):
    T = u.shape[0]
    nx = A.shape[0]
    ny = C.shape[0]
    
    # Forward pass (Kalman Filter)
    x_pred = np.zeros((T + 1, nx, 1))
    P_pred = np.zeros((T + 1, nx, nx))
    x_filt = np.zeros((T + 1, nx, 1))
    P_filt = np.zeros((T + 1, nx, nx))
    
    x_filt[0] = x0
    P_filt[0] = np.eye(nx)
    
    for t in range(T):
        # Prediction
        x_pred[t + 1] = A @ x_filt[t] + B @ u[t] + mu_w
        P_pred[t + 1] = A @ P_filt[t] @ A.T + Sigma_w
        
        # Update
        y_pred = C @ x_pred[t + 1] + mu_v
        S = C @ P_pred[t + 1] @ C.T + Sigma_v
        K = P_pred[t + 1] @ C.T @ np.linalg.inv(S)
        x_filt[t + 1] = x_pred[t + 1] + K @ (y[t + 1] - y_pred)
        P_filt[t + 1] = P_pred[t + 1] - K @ C @ P_pred[t + 1]
    
    # Backward pass (Kalman Smoother)
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    P_cross = np.zeros((T, nx, nx))
    
    for t in range(T - 1, -1, -1):
        A_hat = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + A_hat @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + A_hat @ (P_smooth[t + 1] - P_pred[t + 1]) @ A_hat.T
        P_cross[t] = P_smooth[t + 1] @ A_hat.T
    
    return x_filt, P_filt, x_smooth, P_smooth, P_cross
def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T):
    
    lambda_ = 10
    seed = 2024 # Random seed
    # --- Parameter for DRLQC --- #
    tol = 1e-3
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
    B = C = Q = R = Qf = np.eye(10) 
    #----------------------------
    # You can change theta_v list and lambda_list ! but you also need to change lists at plot_params4_drlqc_nonzeromean.py to get proper plot
    
    if dist=='normal':
        theta_v_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        theta_w_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        #theta_v_list = [1.0, 2.0] # radius of noise ambiguity set
        #theta_w_list = [1.0, 2.0] # radius of noise ambiguity set
    else:
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] # radius of noise ambiguity set
        theta_w_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] # radius of noise ambiguity set
        #theta_v_list = [1.0] # radius of noise ambiguity set
        #theta_w_list = [5.0] # radius of noise ambiguity set
    lambda_list = [3] # disturbance distribution penalty parameter # not used if use_lambda = False
    theta_w = 1.0 # will not be used if use_lambda = True
    #num_x0_samples = 15 #  N_x0 
    theta_x0 = 0.5 # radius of initial state ambiguity set
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
        mu_w = 1.0*np.ones((nx, 1))
        Sigma_w= 0.5*np.eye(nx)
        #initial state distribution parameters
        x0_max = None
        x0_min = None
        x0_mean = 0.5*np.ones((nx,1))
        x0_cov = 0.001*np.eye(nx)
    elif dist == "quadratic":
        #disturbance distribution parameters
        w_max = 0.5*np.ones(nx)
        w_min = -1.5*np.ones(nx)
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
        v_max = 3.0*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
    
    x0 = x0_mean    
    N=50
    # -------Estimate the nominal distribution-------
    # Initialize lists to store data for all sequences
    x_list = []
    y_list = []
    u_list = []

    # Generate N sequences of data with known initial state x[0]
    for i in range(N):
        # Initialize state, input, and output arrays for sequence i
        x = np.zeros((T + 1, nx, 1))
        y = np.zeros((T + 1, ny, 1))
        u = np.zeros((T, nu, 1))

        # Set initial state x[0] (known)
        x[0] = x0

        # Generate input-output data over time horizon T
        for t in range(T):
            # Sample  w_t
            if dist=="normal":
                true_w = normal(mu_w, Sigma_w)
            elif dist=="quadratic":
                true_w = quadratic(w_max, w_min)
                
            # Sample v_t
            if noise_dist=="normal":
                true_v = normal(mu_v, M)
            elif noise_dist=="quadratic":
                true_v = quadratic(v_max, v_min)
                
            # Sample control input u_t from zero-mean Gaussian distribution
            u[t] = np.random.multivariate_normal(np.zeros(nu), np.eye(nu)).reshape(nu, 1)
            # Update state x_{t+1}
            x[t + 1] = A @ x[t] + B @ u[t] + true_w
            # Generate measurement y_{t}
            y[t] = C @ x[t] + true_v 

        # Generate measurement y[T] at final state x[T]
        true_v_T = np.random.multivariate_normal(mu_v.flatten(), M).reshape(ny, 1)
        y[T] = C @ x[T] + true_v_T

        # Append sequence data to lists
        x_list.append(x)
        y_list.append(y)
        u_list.append(u)

    # --- Estimation Procedure ---
    # EM Algorithm Implementation with Kalman Smoother
    max_iterations = 50
    tolerance = 1e-3
    #Initial guess
    mu_w_hat = np.zeros((nx, 1))
    Sigma_w_hat = np.eye(nx)
    mu_v_hat = np.zeros((ny, 1))
    Sigma_v_hat = np.eye(ny)
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        
        # Store previous estimates
        mu_w_prev = mu_w_hat.copy()
        Sigma_w_prev = Sigma_w_hat.copy()
        mu_v_prev = mu_v_hat.copy()
        Sigma_v_prev = Sigma_v_hat.copy()
        
        # Initialize accumulators
        sum_E_w = np.zeros((nx, 1))
        sum_E_ww = np.zeros((nx, nx))
        sum_E_v = np.zeros((ny, 1))
        sum_E_vv = np.zeros((ny, ny))
        total_w_samples = 0
        total_v_samples = 0
        
        for i in range(N):
            u = u_list[i]
            y = y_list[i]
            
            # Run Kalman Smoother
            x_filt, P_filt, x_smooth, P_smooth, P_cross = kalman_smoother(u, y, A, B, C, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, x0)
            
            # M-Step calculations
            T_seq = u.shape[0]
            
            # Process noise expectations
            for t in range(T_seq):
                E_w_t = x_smooth[t + 1] - A @ x_smooth[t] - B @ u[t] - mu_w_hat
                sum_E_w += E_w_t
                total_w_samples += 1
                
                E_ww_t = P_smooth[t + 1] + x_smooth[t + 1] @ x_smooth[t + 1].T - (A @ P_smooth[t] + x_smooth[t + 1] @ x_smooth[t].T @ A.T + B @ u[t] @ x_smooth[t].T @ A.T + A @ x_smooth[t] @ u[t].T @ B.T + B @ u[t] @ u[t].T @ B.T)
                sum_E_ww += E_ww_t
                
            # Measurement noise expectations
            for t in range(T_seq + 1):
                E_v_t = y[t] - C @ x_smooth[t] - mu_v_hat
                sum_E_v += E_v_t
                total_v_samples += 1
                
                E_vv_t = E_v_t @ E_v_t.T + C @ P_smooth[t] @ C.T
                sum_E_vv += E_vv_t
        
        # Update means
        mu_w_hat = sum_E_w / total_w_samples
        mu_v_hat = sum_E_v / total_v_samples
        
        # Update covariances
        Sigma_w_hat = sum_E_ww / total_w_samples
        Sigma_v_hat = sum_E_vv / total_v_samples
        
        # Ensure covariance matrices are symmetric
        Sigma_w_hat = (Sigma_w_hat + Sigma_w_hat.T) / 2
        Sigma_v_hat = (Sigma_v_hat + Sigma_v_hat.T) / 2
        
        # Check for convergence
        delta_mu_w = np.linalg.norm(mu_w_hat - mu_w_prev)
        delta_Sigma_w = np.linalg.norm(Sigma_w_hat - Sigma_w_prev, 'fro')
        delta_mu_v = np.linalg.norm(mu_v_hat - mu_v_prev)
        delta_Sigma_v = np.linalg.norm(Sigma_v_hat - Sigma_v_prev, 'fro')
        
        print(f"Change in mu_w: {delta_mu_w}")
        print(f"Change in Sigma_w: {delta_Sigma_w}")
        print(f"Change in mu_v: {delta_mu_v}")
        print(f"Change in Sigma_v: {delta_Sigma_v}\n")
        
        if (delta_mu_w < tolerance and delta_Sigma_w < tolerance and
            delta_mu_v < tolerance and delta_Sigma_v < tolerance):
            print("Convergence achieved!")
            break
    # Compute errors in norms for means
    error_mu_w = np.linalg.norm(mu_w_hat - mu_w)
    error_mu_v = np.linalg.norm(mu_v_hat - mu_v)

    # Compute errors in norms for covariances
    error_Sigma_w = np.linalg.norm(Sigma_w_hat - Sigma_w, 'fro')
    M_hat = Sigma_v_hat
    error_M = np.linalg.norm(M_hat - M, 'fro')
    
    print("\nError Norm for mu_w (||mu_w_hat - mu_w||):", error_mu_w)

    print("Error Norm for Sigma_w (Frobenius norm):", error_Sigma_w)

    print("\nError Norm for mu_v (||mu_v_hat - mu_v||):", error_mu_v)

    print("Error Norm for Sigma_v (Frobenius norm):", error_M)
    
    
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
                path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/ioEM/".format(dist, noise_dist)
            else:
                path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/ioEM/".format(dist, noise_dist)
                
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
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/ioEM/raw/".format(dist, noise_dist)
            else:
                rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/ioEM/raw/".format(dist, noise_dist)
                
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
        path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/io/".format(dist, noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/io/".format(dist, noise_dist)
        save_data(path + 'nonzero_wdrc_lambda.pkl',WDRC_lambda)
        save_data(path + 'nonzero_drce_lambda.pkl',DRCE_lambda)
        
    
            
    print("Params data generation Completed !")
    print("Please make sure your lambda_list(or theta_w_list) and theta_v_list plot_params4_usingio.py is as desired")
    if use_lambda:
        print("Now use : python plot_params4_usingio.py --use_lambda --dist "+ dist + " --noise_dist " + noise_dist)
    else:
        print("Now use : python plot_params4_usingio.py --dist "+ dist + " --noise_dist " + noise_dist)
    
            

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
