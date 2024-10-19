# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:22:07 2024

@author: astgh
"""
import numpy as np
import argparse
   
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
    
    K = np.zeros((T+1, nx, ny))
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
        K_t = P_hat_pred[t] @ C.T @ np.linalg.inv(C @ P_hat_pred[t] @ C.T + Sigma_v_hat + reg_eps*np.eye(ny))  # Kalman gain
        #K[t] = K_t # STORE
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
        J[t] = P_hat[t] @ A.T @ np.linalg.inv(P_hat_pred[t+1])

        # Smoothed state estimate
        x_tilde[t] = x_hat[t] + J[t] @ (x_tilde[t+1] - x_hat_pred[t+1])

        # Smoothed covariance estimate
        P_tilde[t] = P_hat[t] + J[t] @ (P_tilde[t+1] - P_hat_pred[t+1]) @ J[t].T
        if np.any(np.linalg.eigvals(P_tilde[t]) <= 0):
            print(f'(KS) Non PSD covariance at time {t}!', np.min(np.linalg.eigvals(P_tilde[t])))
            P_tilde[t] += (-np.min(np.linalg.eigvals(P_tilde[t])) + reg_eps)*np.eye(nx)

    V_tilde[T-1] = (np.eye(nx) - K @ C ) @ A @ P_tilde[T-1] #V_{T,T-1}
    #V_tilde[T-1] = (np.eye(nx) - K[T] @ C) @ A @ P_tilde[T-1]
    for t in range(T-2, -1, -1):
        # Calculate the Lag-1 autocovariance of the smoothed state
        V_tilde[t] = P_tilde[t+1] @ J[t].T + J[t+1] @ (V_tilde[t+1] - A @ P_tilde[t+1]) @ J[t].T #V_{t+1,t}
        #V_tilde[t] = P_hat[t] @ J[t].T + J[t+1] @ (V_tilde[t+1] - A @ P_tilde[t+1]) @ J[t].T

    return x_tilde, P_tilde, V_tilde


def log_likelihood(y_all, A, C, x_tilde, P_tilde, V_tilde, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat):
    nx = mu_x0_hat.shape[0]  # State dimension
    ny = mu_v_hat.shape[0]  # Measurement dimension
    T = x_tilde.shape[0]-1
    
    Sigma_w_hat_inv = np.linalg.inv(Sigma_w_hat + reg_eps*np.eye(nx))
    Sigma_v_hat_inv = np.linalg.inv(Sigma_v_hat + reg_eps*np.eye(ny))
    Sigma_x0_hat_inv = np.linalg.inv(Sigma_x0_hat + reg_eps*np.eye(nx))
    c = - 0.5*(T*(nx + ny) + ny)*np.log(2*np.pi) - T/2*np.log(np.linalg.det(Sigma_v_hat) + reg_eps) - T/2*np.log(np.linalg.det(Sigma_w_hat) + reg_eps) - 0.5*np.log(np.linalg.det(Sigma_x0_hat) + reg_eps) - 0.5*np.trace(Sigma_x0_hat_inv @ P_tilde[0]) - 0.5*(x_tilde[0] - mu_x0_hat).T @ Sigma_x0_hat_inv @ (x_tilde[0] - mu_x0_hat)

    for t in range(T):
        c += -0.5*( (y_all[t] - C @ x_tilde[t+1] - mu_v_hat).T @ Sigma_v_hat_inv @ (y_all[t] - C @ x_tilde[t+1] - mu_v_hat)  \
                  + (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat).T @ Sigma_w_hat_inv @ (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat) \
                  + np.trace(C.T @ Sigma_v_hat_inv @ C @ P_tilde[t+1]) \
                  + np.trace(Sigma_w_hat_inv @ P_tilde[t+1]) \
                  - 2 * np.trace(Sigma_w_hat_inv @ A @ V_tilde[t].T) )
        
    return c
                    
    
def log_lh_max(x_tilde, P_tilde, V_tilde, A, C, y_all):



    nx = A.shape[0]  # State dimension
    ny = C.shape[0]  # Measurement dimension
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
        # Process noise covariance update
        delta_w = x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new
        Sigma_w_hat_new += (delta_w @ delta_w.T +
                            P_tilde[t+1] +
                            A @ P_tilde[t] @ A.T -
                            A @ V_tilde[t].T -
                            V_tilde[t] @ A.T)

        # Measurement noise covariance update
        delta_v = y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new
        Sigma_v_hat_new += (delta_v @ delta_v.T +
                            C @ P_tilde[t+1] @ C.T)
        # Sigma_w_hat_new += (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new) @ (x_tilde[t+1] - A @ x_tilde[t] - mu_w_hat_new).T + \
        #                    A @ P_tilde[t] @ A.T + P_tilde[t+1] - 2 * V_tilde[t] @ A.T
        # Sigma_v_hat_new += (y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new) @ (y_all[t] - C @ x_tilde[t+1] - mu_v_hat_new).T - C @ P_tilde[t+1] @ C.T
    
    
    Sigma_w_hat_new = (1/T)*Sigma_w_hat_new + reg_eps * np.eye(nx)
    Sigma_v_hat_new = (1/T)*Sigma_v_hat_new + reg_eps * np.eye(ny)
        
        
    return mu_x0_hat_new, mu_w_hat_new, mu_v_hat_new, Sigma_x0_hat_new, Sigma_w_hat_new, Sigma_v_hat_new
        
def main(dist, noise_dist, num_sim, num_samples, num_noise_samples, T):


    #-------Initialization-------
    nx = 10 #state dimension
    nu = 10 #control input dimension
    ny = 10#output dimension
    temp = np.ones((nx, nx))
    A = 0.2*(np.eye(nx) + np.triu(temp, 1) - np.triu(temp, 2))
    B= np.eye(10)
    C = Q = R = Qf = np.eye(10) 
    
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
        M = 1.0*np.eye(ny) #observation noise covariance
        mu_v = 0.1*np.ones((ny, 1))
    elif noise_dist =="quadratic":
        v_min = -1.5*np.ones(ny)
        v_max = 2.5*np.ones(ny)
        mu_v = (0.5*(v_max + v_min))[..., np.newaxis]
        M = 3.0/20.0 *np.diag((v_max-v_min)**2) #observation noise covariance
    
    print(f'real data: mu_w: {mu_w}, mu_v: {mu_v}, Sigma_w: {Sigma_w}, Sigma_v: {M}')
    T = 1000
    x_all, y_all = generate_data(T, nx, ny, nu, A, B, C, mu_w, Sigma_w, mu_v, M, x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist)
    
    eps_log = 1e-6
    eps_param = 1e-6
    


    # -------Estimate the nominal distribution-------
    # Initialize estimates
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov
    
    
    
    max_iter = 2000
    
    x_hat = np.zeros((max_iter, T+1, nx, 1))
    P_hat = np.zeros((max_iter, T+1, nx, nx))
    x_hat_pred = np.zeros((max_iter, T+1, nx, 1))
    P_hat_pred = np.zeros((max_iter, T+1, nx, nx))
    x_tilde = np.zeros((max_iter, T+1, nx, 1))
    P_tilde = np.zeros((max_iter, T+1, nx, nx))
    V_tilde = np.zeros((max_iter, T, nx, nx))      
    
    log_lh = np.zeros(max_iter+1)
    
    for i in range(max_iter):
        print(f'\n--------Iteration {i}----------')
        #---------E-Step------------
        x_hat[i], P_hat[i], x_hat_pred[i], P_hat_pred[i], K = kalman_filter(A, B, C, mu_x0_hat, Sigma_x0_hat, mu_w_hat, Sigma_w_hat, mu_v_hat, Sigma_v_hat, y_all, T)
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
