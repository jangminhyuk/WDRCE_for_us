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
reg_eps = 1e-6

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
# Function to generate the true states and measurements (data generation)
def generate_data(T, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    u = np.zeros((T, nu, 1))
    x_true_all = np.zeros((T + 1, nx, 1))  # True state sequences
    y_all = np.zeros((T, ny, 1))  # Measurements for each sequence
    
    # Initialize the true state for each sequence
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)  # initial true state
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)  # initial true state
    
    x_true_all[0] = x_true  # Set initial state
    
    for t in range(T):
        # Sample true process noise and measurement noise
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)

        # True state update (process model)
        x_true = A @ x_true + B @ u[t] + true_w  # true x_t+1
        x_true_all[t + 1] = x_true

        # Measurement (observation model)
        y_t = C @ x_true + true_v #Note that y's index is shifted, i.e. y[0] correspondst to y_1
        y_all[t] = y_t

    return x_true_all, y_all


def kalman_filter(A, B, C, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, y, T):
      # Get the dimensions
    nx = A.shape[0]  # State dimension
    ny = C.shape[0]  # Measurement dimension

    # Initialize the arrays for storing results
    x_hat = np.zeros((T+1, nx, 1))  # Filtered state estimates
    P_hat = np.zeros((T+1, nx, nx))  # Filtered covariance estimates
    x_hat_pred = np.zeros((T+1, nx, 1))  # Predicted state estimates
    P_hat_pred = np.zeros((T+1, nx, nx))  # Predicted covariance estimates

    # Set initial state estimates
    x_hat[0] = mu_x0_hat
    P_hat[0] = Sigma_x0_hat

    # Kalman filter algorithm
    for t in range(1,T+1):
        # 1. Prediction step
        x_hat_pred[t] = A @ x_hat[t-1] + mu_w_hat
        P_hat_pred[t] = A @ P_hat[t-1] @ A.T + Sigma_w_hat

        if np.any(np.linalg.eigvals(P_hat_pred[t]) <= 0):
            print(f'(KF_ Non PSD pred covariance at time {t}!', np.min(np.linalg.eigvals(P_hat_pred[t])))
            P_hat_pred[t] += (-np.min(np.linalg.eigvals(P_hat_pred[t])) + reg_eps)*np.eye(nx)


        # 2. Update step
        K_t = np.linalg.solve(C @ P_hat_pred[t] @ C.T + Sigma_v_hat + reg_eps*np.eye(ny), P_hat_pred[t] @ C.T)  # Kalman gain
        x_hat[t] = x_hat_pred[t] + K_t @ (y[t-1] - C @ x_hat_pred[t] - mu_v_hat)
        P_hat[t] = (np.eye(nx) - K_t @ C) @ P_hat_pred[t]
        
        if np.any(np.linalg.eigvals(P_hat[t]) <= 0):
            print(f'(KF_ Non PSD covariance at time {t}!', np.min(np.linalg.eigvals(P_hat[t])))
            P_hat[t] += (-np.min(np.linalg.eigvals(P_hat[t])) + reg_eps)*np.eye(nx)

    return x_hat, P_hat, x_hat_pred, P_hat_pred, K_t


def kalman_smoother(x_hat, P_hat, x_hat_pred, P_hat_pred, A, B, C, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, K):
    # Get the state dimension
    nx = A.shape[0]
    T = x_hat.shape[0]-1
    
    # Initialize arrays for storing results
    x_tilde = np.zeros((T+1, nx, 1))  # Smoothed state estimates
    P_tilde = np.zeros((T+1, nx, nx))  # Smoothed covariance estimates
    V_tilde = np.zeros((T, nx, nx))    # Lag-1 autocovariance of smoothed state estimates
    J = np.zeros((T, nx, nx)) 
    
    # The final smoothed estimate is the same as the final filtered estimate
    x_tilde[T] = x_hat[T]
    P_tilde[T] = P_hat[T]

    # Kalman smoother algorithm (backward pass)
    for t in range(T-1, -1, -1):
        # Calculate the smoother gain
        J[t] = np.linalg.solve(P_hat_pred[t+1], P_hat[t] @ A.T)

        # Smoothed state estimate
        x_tilde[t] = x_hat[t] + J[t] @ (x_tilde[t+1] - x_hat_pred[t+1])

        # Smoothed covariance estimate
        P_tilde[t] = P_hat[t] + J[t] @ (P_tilde[t+1] - P_hat_pred[t+1]) @ J[t].T
        if np.any(np.linalg.eigvals(P_tilde[t]) <= 0):
            print(f'(KS) Non PSD covariance at time {t}!', np.min(np.linalg.eigvals(P_tilde[t])))
            P_tilde[t] += (-np.min(np.linalg.eigvals(P_tilde[t])) + reg_eps)*np.eye(nx)

    V_tilde[T-1] = (np.eye(nx) - K @ C ) @ A @ P_tilde[T-1] #V_{T,T-1}
    for t in range(T-2, 1, -1):
        # Calculate the Lag-1 autocovariance of the smoothed state
        V_tilde[t] = P_tilde[t+1] @ J[t].T + J[t+1] @ (V_tilde[t+1] - A @ P_tilde[t+1]) @ J[t].T #V_{t+1,t}

    return x_tilde, P_tilde, V_tilde

    
def log_likelihood(y_all, A, C, x_tilde, P_tilde, V_tilde, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat):
    nx = mu_x0_hat.shape[0]  # State dimension
    ny = mu_v_hat.shape[0]  # Measurement dimension
    T = x_tilde.shape[0]-1
    
    Sigma_w_hat_inv = np.linalg.inv(Sigma_w_hat + reg_eps*np.eye(nx))
    Sigma_v_hat_inv = np.linalg.inv(Sigma_v_hat + reg_eps*np.eye(ny))
    Sigma_x0_hat_inv = np.linalg.inv(Sigma_x0_hat + reg_eps*np.eye(nx))
    c = - 0.5*(T*(nx + ny) + nx)*np.log(2*np.pi) - T/2*np.log(np.linalg.det(Sigma_v_hat) + reg_eps) - T/2*np.log(np.linalg.det(Sigma_w_hat) + reg_eps) - 0.5*np.log(np.linalg.det(Sigma_x0_hat) + reg_eps) - 0.5*np.trace(Sigma_x0_hat_inv @ P_tilde[0]) - 0.5*(x_tilde[0] - mu_x0_hat).T @ Sigma_x0_hat_inv @ (x_tilde[0] - mu_x0_hat)

    for t in range(T):
        c += -0.5*( (y_all[t] - C @ x_tilde[t+1] - mu_v_hat).T @ Sigma_v_hat_inv @ (y_all[t] - C @ x_tilde[t+1] - mu_v_hat)  \
                  + (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat).T @ Sigma_w_hat_inv @ (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat) \
                  + np.trace(C.T @ Sigma_v_hat_inv @ C @ P_tilde[t+1]) \
                  + np.trace(Sigma_w_hat_inv @ P_tilde[t+1]) \
                  - 2 * np.trace(Sigma_w_hat_inv @ A @ V_tilde[t].T) )\
        
    return c
    

def log_lh_max(x_tilde, P_tilde, V_tilde, A, C, y_all):

    nx = A.shape[0]  # State dimension
    ny = C.shape[0]  # Measurement dimension
    #print(x_tilde.shape[0])
    T = x_tilde.shape[0]-1
    
    mu_x0_hat_new = x_tilde[0] 
    Sigma_x0_hat_new = P_tilde[0] + reg_eps * np.eye(nx)
    
    mu_w_hat_new = np.zeros((nx, 1))
    mu_v_hat_new = np.zeros((ny, 1))
    Sigma_w_hat_new = np.zeros((nx, nx)) 
    Sigma_v_hat_new = np.zeros((ny, ny))
    
    for t in range(T): 
        mu_w_hat_new += x_tilde[t+1] - A @ x_tilde[t]         
        mu_v_hat_new += y_all[t] - C @ x_tilde[t+1]
        
    mu_w_hat_new = 1/T*mu_w_hat_new
    mu_v_hat_new = 1/T*mu_v_hat_new
    
    for t in range(T):
        #print(x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new, y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new)
        Sigma_w_hat_new += (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new) @ (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new).T + \
                           A @ P_tilde[t] @ A.T + P_tilde[t+1] - 2 * A @ V_tilde[t].T
        Sigma_v_hat_new += (y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new) @ (y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new).T + C @ P_tilde[t+1] @ C.T
    
    Sigma_w_hat_new = 1/T*Sigma_w_hat_new + reg_eps * np.eye(nx)
    Sigma_v_hat_new = 1/T*Sigma_v_hat_new + reg_eps * np.eye(ny)
        
        
    return mu_x0_hat_new, mu_w_hat_new, mu_v_hat_new, Sigma_x0_hat_new, Sigma_w_hat_new, Sigma_v_hat_new
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
        mu_w = 0.1*np.ones((nx, 1))
        Sigma_w= 1.5*np.eye(nx)
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
        M = 2.0*np.eye(ny) #observation noise covariance
        mu_v = 0.2*np.ones((ny, 1))
    elif noise_dist =="quadratic":
        v_min = -1.5*np.ones(ny)
        v_max = 2.5*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
    #x0 = x0_mean
    print(f'real data: \n mu_w: {mu_w}, \n mu_v: {mu_v}, \n Sigma_w: {Sigma_w}, \n Sigma_v: {M}')
    
    
    N = 1000 # horizon for data generation
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist)
    
    eps_log = 1e-6
    eps_param = 1e-6
    
    
    # -------Estimate the nominal distribution-------
    # Initialize estimates
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    # save best
    mu_w_hat_best = np.zeros((nx, 1))
    mu_v_hat_best = np.zeros((ny, 1))
    mu_x0_hat_best = x0_mean.copy()
    Sigma_w_hat_best = np.eye(nx)
    Sigma_v_hat_best = np.eye(ny)
    Sigma_x0_hat_best = x0_cov.copy()
    log_lh_best = -np.inf
    
    
    max_iter = 1000
    
    x_hat = np.zeros((max_iter, N+1, nx, 1))
    P_hat = np.zeros((max_iter, N+1, nx, nx))
    x_hat_pred = np.zeros((max_iter, N+1, nx, 1))
    P_hat_pred = np.zeros((max_iter, N+1, nx, nx))
    x_tilde = np.zeros((max_iter, N+1, nx, 1))
    P_tilde = np.zeros((max_iter, N+1, nx, nx))
    V_tilde = np.zeros((max_iter, N, nx, nx))      
    
    log_lh = np.zeros(max_iter+1)


    for i in range(max_iter):
        print(f'\n--------Iteration {i}----------')
        #---------E-Step------------
        x_hat[i], P_hat[i], x_hat_pred[i], P_hat_pred[i], K = kalman_filter(A, B, C, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, y_all, N)   
        x_tilde[i], P_tilde[i], V_tilde[i] = kalman_smoother(x_hat[i], P_hat[i], x_hat_pred[i], P_hat_pred[i], A, B, C, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, K)
    
        log_lh[i] = log_likelihood(y_all, A, C, x_tilde[i], P_tilde[i], V_tilde[i], mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat)
        
        
        #---------M-Step------------
        mu_x0_hat_new, mu_w_hat_new, mu_v_hat_new, Sigma_x0_hat_new, Sigma_w_hat_new, Sigma_v_hat_new = log_lh_max(x_tilde[i], P_tilde[i], V_tilde[i], A, C, y_all)


        Sigma_w_hat = Sigma_w_hat_new.copy()
        Sigma_v_hat = Sigma_v_hat_new.copy()
        Sigma_x0_hat = Sigma_x0_hat_new.copy()

        
        if np.any(np.linalg.eigvals(Sigma_w_hat) <= 0):
            print(f'Non PSD covariance of w at iter {i}!', np.linalg.det(Sigma_w_hat))
            Sigma_w_hat += -np.min(np.linalg.eigvals(Sigma_w_hat))*np.eye(nx)
                                   
        if np.any(np.linalg.eigvals(Sigma_v_hat) <= 0):
            print(f'Non PSD covariance of v at iter {i}!', np.linalg.det(Sigma_v_hat))
            Sigma_v_hat += -np.min(np.linalg.eigvals(Sigma_v_hat))*np.eye(ny)
                                   
        if np.any(np.linalg.eigvals(Sigma_x0_hat) <= 0):
            print(f'Non PSD covariance of w at iter {i}!', np.linalg.det(Sigma_x0_hat))
            Sigma_x0_hat += -np.min(np.linalg.eigvals(Sigma_x0_hat))*np.eye(nx)
            
        mu_w_hat = mu_w_hat_new.copy()
        mu_v_hat = mu_v_hat_new.copy()
        mu_x0_hat = mu_x0_hat_new.copy()
        
        if log_lh[i]>log_lh_best:
            mu_w_hat_best = mu_w_hat_new.copy()
            mu_v_hat_best = mu_v_hat_new.copy()
            mu_x0_hat_best = x0_mean.copy()
            Sigma_w_hat_best = Sigma_w_hat_new.copy()
            Sigma_v_hat_best = Sigma_v_hat_new.copy()
            Sigma_x0_hat_best = x0_cov.copy()
            log_lh_best = log_lh[i]

        # Mean estimation errors (Euclidean norms)
        error_mu_w = np.linalg.norm(mu_w_hat - mu_w)
        error_mu_v = np.linalg.norm(mu_v_hat - mu_v)
        error_mu_x0 = np.linalg.norm(mu_x0_hat - x0_mean)

        # Covariance estimation errors (Frobenius norms)
        error_Sigma_w = np.linalg.norm(Sigma_w_hat - Sigma_w, 'fro')
        error_Sigma_v = np.linalg.norm(Sigma_v_hat - M, 'fro')
        error_Sigma_x0 = np.linalg.norm(Sigma_x0_hat - x0_cov, 'fro')
        
        
        print("\nEstimation Error (mu_w): {:.6f}".format(error_mu_w))
        print("\nEstimation Error (Sigma_w): {:.6f}".format(error_Sigma_w))
        print("\nEstimation Error (mu_v): {:.6f}".format(error_mu_v))
        print("\nEstimation Error (M): {:.6f}".format(error_Sigma_v))
        print("\nEstimation Error (x0_mean): {:.6f}".format(error_mu_x0))
        print("\nEstimation Error (x0_cov): {:.6f}".format(error_Sigma_x0))
        print("\nLog-Likelihood: {:.6f}".format(log_lh[i]))
        
        params_conv = np.all([error_mu_w <= eps_param, error_mu_v <= eps_param, error_mu_x0 <= eps_param, np.all(error_Sigma_w <= eps_param), np.all(error_Sigma_v <= eps_param), np.all(error_Sigma_x0 <= eps_param)])
        
        if i>0:
            if log_lh[i] - log_lh[i-1] <= eps_log and params_conv:
                print('Converged!')
                break
    #exit()
    # Choose the best one
    print("Nominal distributions are ready")
    mu_w_hat = mu_w_hat_best
    mu_v_hat = mu_v_hat_best
    Sigma_w_hat = Sigma_w_hat_best
    M_hat = Sigma_v_hat_best 
    
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
