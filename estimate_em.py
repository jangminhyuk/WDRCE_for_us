# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:22:07 2024

@author: astgh
"""
import numpy as np
import argparse
from pykalman import KalmanFilter
import matplotlib.pyplot as plt



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


        # Measurement (observation model)
        y_t = C @ x_true + true_v #Note that y's index is shifted, i.e. y[0] correspondst to y_1
        y_all[t] = y_t
        
        # True state update (process model)
        x_true = A @ x_true + B @ u[t] + true_w  # true x_t+1
        x_true_all[t + 1] = x_true


    return x_true_all, y_all



def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T):


    #-------Initialization-------
    nx = 3 #state dimension
    nu = 3 #control input dimension
    ny = 3 #output dimension
    temp = np.ones((nx, nx))
    A = 0.2*(np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B= np.eye(nx)
    C = Q = R = Qf = np.eye(nx) 
    
    #-------Disturbance Distribution-------
    if dist == "normal":
        #disturbance distribution parameters
        w_max = None
        w_min = None
        mu_w = 0.1*np.ones((nx, 1))
        Sigma_w= 0.6*np.eye(nx)
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
        M = 0.2*np.eye(ny) #observation noise covariance
        mu_v = 0.1*np.ones((ny, 1))
    elif noise_dist =="quadratic":
        v_min = -1.5*np.ones(ny)
        v_max = 2.5*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
    
    print(f'real data: mu_w: {mu_w}, mu_v: {mu_v}, Sigma_w: {Sigma_w}, Sigma_v: {M}')
    N = 1000
    x_all, y_all = generate_data(N, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist)
    
    y_all = y_all.squeeze()
    eps_param = 1e-5
    eps_log = 1e-4
    # Initialize estimates
    mu_w_hat = np.zeros(nx)
    mu_v_hat = np.zeros(ny)
    mu_x0_hat = x0_mean.squeeze()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov
    
    kf = KalmanFilter(A, C, Sigma_w_hat, Sigma_v_hat, mu_w_hat, mu_v_hat, mu_x0_hat, Sigma_x0_hat,
                      em_vars=[
                        'transition_covariance', 'observation_covariance',
                        'transition_offsets', 'observation_offsets',
                        #'initial_state_mean', 'initial_state_covariance'
                      ])

    max_iter = 1000
    loglikelihoods = np.zeros(max_iter)
    errors_mu_w = []
    errors_mu_v = []
    errors_mu_x0 = []
    errors_Sigma_w = []
    errors_Sigma_v = []
    errors_Sigma_x0 = []

    
    for i in range(max_iter):
        print(f'------- Iteration {i} ------------')
        kf = kf.em(X=y_all, n_iter=1)
        loglikelihoods[i] = kf.loglikelihood(y_all)
        

        Sigma_w_hat = kf.transition_covariance
        Sigma_v_hat = kf.observation_covariance
        mu_w_hat = kf.transition_offsets
        mu_v_hat = kf.observation_offsets
        mu_x0_hat = kf.initial_state_mean
        Sigma_x0_hat = kf.initial_state_covariance



        # Mean estimation errors (Euclidean norms)
        error_mu_w = np.linalg.norm(mu_w_hat - mu_w)
        error_mu_v = np.linalg.norm(mu_v_hat - mu_v)
        error_mu_x0 = np.linalg.norm(mu_x0_hat - x0_mean)

        # Covariance estimation errors (Frobenius norms)
        error_Sigma_w = np.linalg.norm(Sigma_w_hat - Sigma_w, 'fro')
        error_Sigma_v = np.linalg.norm(Sigma_v_hat - M, 'fro')
        error_Sigma_x0 = np.linalg.norm(Sigma_x0_hat - x0_cov, 'fro')
        
                
        # Store errors for plotting
        errors_mu_w.append(error_mu_w)
        errors_mu_v.append(error_mu_v)
        errors_mu_x0.append(error_mu_x0)
        errors_Sigma_w.append(error_Sigma_w)
        errors_Sigma_v.append(error_Sigma_v)
        errors_Sigma_x0.append(error_Sigma_x0)

        
        
        print("\nEstimation Error (mu_w): {:.6f}".format(error_mu_w))
        print("\nEstimation Error (Sigma_w): {:.6f}".format(error_Sigma_w))
        print("\nEstimation Error (mu_v): {:.6f}".format(error_mu_v))
        print("\nEstimation Error (M): {:.6f}".format(error_Sigma_v))
        print("\nEstimation Error (x0_mean): {:.6f}".format(error_mu_x0))
        print("\nEstimation Error (x0_cov): {:.6f}".format(error_Sigma_x0))
        print("\nLog-Likelihood: {:.6f}".format(loglikelihoods[i]))
        
        params_conv = np.all([error_mu_w <= eps_param, error_mu_v <= eps_param, error_mu_x0 <= eps_param, np.all(error_Sigma_w <= eps_param), np.all(error_Sigma_v <= eps_param), np.all(error_Sigma_x0 <= eps_param)])
        
        if i>0:
            if loglikelihoods[i] - loglikelihoods[i-1] <= eps_log and params_conv:
                print('Converged!')
                break
    
        # -------- Plotting --------
    iterations = range(i + 1)  # Number of iterations

    # Plot estimation errors
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(iterations, errors_mu_w, label='Error in mu_w')
    plt.plot(iterations, errors_mu_v, label='Error in mu_v')
    plt.plot(iterations, errors_mu_x0, label='Error in mu_x0')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Errors')
    plt.legend()
    plt.title('Mean Estimation Errors over Iterations')

    plt.subplot(2, 1, 2)
    plt.plot(iterations, errors_Sigma_w, label='Error in Sigma_w')
    plt.plot(iterations, errors_Sigma_v, label='Error in Sigma_v')
    plt.plot(iterations, errors_Sigma_x0, label='Error in Sigma_x0')
    plt.xlabel('Iterations')
    plt.ylabel('Covariance Errors')
    plt.legend()
    plt.title('Covariance Estimation Errors over Iterations')

    plt.tight_layout()
    plt.show(block=False)

    # Plot log-likelihood
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, loglikelihoods)
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood over Iterations')
    plt.grid(True)
    plt.show()        


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

