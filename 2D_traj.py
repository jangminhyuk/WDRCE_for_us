import numpy as np
import matplotlib.pyplot as plt
import argparse
from controllers.DRCE import DRCE
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
    elif dist=="quadratic":
        w = quadratic(w_max, w_min, N=N_sample)

    mean_ = np.average(w, axis = 1)
    diff = (w.T - mean_)[...,np.newaxis]
    var_ = np.average( (diff @ np.transpose(diff, (0,2,1))) , axis = 0)
    return np.tile(mean_[...,np.newaxis], (T, 1, 1)), np.tile(var_, (T, 1, 1))

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Select trajectory type for simulation.')
parser.add_argument('--trajectory', type=str, choices=['curvy', 'circular'], default='curvy',
                    help='Choose the trajectory type: curvy or circular.')
args = parser.parse_args()

# Constants
A = 5.0  # Amplitude of the sine wave (for curvy trajectory)
B = 1.0  # Slope of the linear y trajectory (for curvy trajectory)
r = 5.0  # Radius of the circle (for circular trajectory)
omega = 0.5  # Angular frequency

# Time for simulation
T = 20  # Simulation time
dt = 0.1  # Time step
N = int(T / dt)  # Number of steps (finite horizon)

# Discrete-time system dynamics (for the 4D double integrator: x, y positions and velocities)
A_d = np.array([[1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]])

B_d = np.array([[0.5 * dt**2, 0],
                [dt, 0],
                [0, 0.5 * dt**2],
                [0, dt]])

# LQR weighting matrices
Q = np.eye(4) * np.array([10, 1, 10, 1])  # Heavily penalize position errors, less for velocity errors
R = 0.01 * np.eye(2)  # Control cost for both x and y accelerations

# Kalman filter parameters
process_noise_cov = np.eye(4) * 0.01  # Process noise covariance
measurement_noise_cov = np.eye(2) * 0.1  # Measurement noise covariance
C = np.array([[0, 1, 0, 0],  # We measure only x and y positions
              [0, 0, 0, 1]])

# Process and measurement noise means
process_noise_mean = np.array([0.01, 0.01, 0.01, 0.01])  # Mean of process noise
measurement_noise_mean = np.array([0.1, 0.1])  # Mean of measurement noise

dist = "normal"
noise_dist = "normal"
nx = 4
nu = 2
ny = 2
num_x0_samples = 10
num_samples = 10
num_noise = 10

#-------Estimate the nominal distribution-------
# Nominal initial state distribution
x0_mean_hat, x0_cov_hat = gen_sample_dist(dist, 1, num_x0_samples, mu_w=x0_mean, Sigma_w=x0_cov, w_max=x0_max, w_min=x0_min)
# Nominal Disturbance distribution
mu_hat, Sigma_hat = gen_sample_dist(dist, T+1, num_samples, mu_w=mu_w, Sigma_w=Sigma_w, w_max=w_max, w_min=w_min)
# Nominal Noise distribution
v_mean_hat, M_hat = gen_sample_dist(noise_dist, T+1, num_noise, mu_w=mu_v, Sigma_w=M, w_max=v_max, w_min=v_min)

M_hat = M_hat + 1e-8*np.eye(ny) # to prevent numerical error from inverse in standard KF at small sample size
Sigma_hat = Sigma_hat + 1e-8*np.eye(nx)
x0_cov_hat = x0_cov_hat + 1e-8*np.eye(nx) 

# Desired trajectory function based on the command-line argument
def desired_trajectory(t):
    if args.trajectory == 'curvy':
        # Curvy trajectory: sine wave for x and linear ramp for y
        x_d = A * np.sin(omega * t)  # Sine wave for x
        vx_d = A * omega * np.cos(omega * t)  # Velocity for x
        y_d = B * t  # Linear ramp for y
        vy_d = B  # Constant velocity in y
    elif args.trajectory == 'circular':
        # Circular trajectory: circular motion for both x and y
        x_d = r * np.cos(omega * t)
        vx_d = -r * omega * np.sin(omega * t)
        y_d = r * np.sin(omega * t)
        vy_d = r * omega * np.cos(omega * t)
    return np.array([x_d, vx_d, y_d, vy_d])

# Set the initial state to match the first point of the desired trajectory
initial_trajectory = desired_trajectory(0)  # Get the initial desired state at t=0
initial_position_velocity = initial_trajectory  # This becomes the initial state of the system

# Finite-horizon LQR: Backward recursion to compute gains for each step
def finite_horizon_lqr(A_d, B_d, Q, R, N):
    P = np.zeros((N+1, A_d.shape[0], A_d.shape[1]))
    K = np.zeros((N, B_d.shape[1], A_d.shape[0]))
    
    # Initialize with terminal cost
    P[N] = Q
    
    # Backward recursion for K and P
    for i in range(N-1, -1, -1):
        P_dot = A_d.T @ P[i+1] @ A_d - (A_d.T @ P[i+1] @ B_d @ np.linalg.inv(R + B_d.T @ P[i+1] @ B_d) @ B_d.T @ P[i+1] @ A_d) + Q
        P[i] = P_dot
        K[i] = np.linalg.inv(R + B_d.T @ P[i+1] @ B_d) @ B_d.T @ P[i+1] @ A_d

    return K

# Generate finite-horizon LQR gains
K = finite_horizon_lqr(A_d, B_d, Q, R, N)

# Simulation variables
time = np.arange(0, T, dt)
state = np.zeros((len(time), 4))  # State [x, vx, y, vy]
estimated_state = np.zeros((len(time), 4))  # Estimated state from Kalman filter

# Initialize the state and Kalman filter estimate with the initial position and velocity from the trajectory
state[0, :] = initial_position_velocity  # Set the initial state to match the trajectory at t=0
x_hat = initial_position_velocity.reshape(4, 1)  # Initial estimate as 4D vector
P_kf = np.eye(4)  # Initial error covariance

# Initialize DRCE controller
lambda_ = 100
theta_w = 0.5
theta = 0.5 # theta_v
theta_x0 = 0.5

system_data = (A_d, B_d, C, Q, Q, R, measurement_noise_cov)
#drce = DRCE(lambda_, theta_w, theta, theta_x0, T, dist, noise_dist, system_data, mu_w_hat, Sigma_w_hat, x0_mean, x0_cov, x0_max, x0_min, mu_w, Sigma_w, w_max, w_min, v_max, v_min, mu_v, mu_v_hat, M_hat, x0_mean_hat, x0_cov_hat, use_lambda, use_optimal_lambda)
            


# Simulation loop
for i in range(len(time) - 1):
    t = time[i]

    # Get desired trajectory at current time step (desired position and velocity)
    traj = desired_trajectory(t).reshape(4, 1)
    error = state[i, :].reshape(4, 1) - traj  # Error as a 4D vector

    # Compute control input using finite-horizon LQR gain for the current step
    u = -K[i] @ error  # Control input `u` as a 2D vector (shape `(2, 1)`)

    # Update the state using the discrete-time system dynamics, with process noise
    process_noise = np.random.normal(process_noise_mean, np.sqrt(np.diag(process_noise_cov)), 4).reshape(4, 1)
    state[i+1, :] = (A_d @ state[i, :].reshape(4, 1) + B_d @ u + process_noise).flatten()  # Flatten for storing

    # Kalman filter update
    # Predict
    x_hat = A_d @ x_hat + B_d @ u + process_noise  # Updated x_hat as 4D vector
    P_kf = A_d @ P_kf @ A_d.T + process_noise_cov

    # Update with measurement (noisy x and y positions)
    measurement_noise = np.random.normal(measurement_noise_mean, np.sqrt(np.diag(measurement_noise_cov)), 2)
    y_measured = state[i+1, [0, 2]] + measurement_noise  # Simulated noisy measurement
    innovation = y_measured.reshape(2, 1) - C @ x_hat
    S = C @ P_kf @ C.T + measurement_noise_cov
    K_kf = P_kf @ C.T @ np.linalg.inv(S)
    x_hat = x_hat + K_kf @ innovation  # Ensure x_hat remains a 4D vector
    P_kf = (np.eye(4) - K_kf @ C) @ P_kf

    # Store estimated state
    estimated_state[i+1, :] = x_hat.flatten()  # Flatten to store in 2D array

# Plotting the results with improved visualization
plt.figure(figsize=(10, 6))

# Plot the actual tracked trajectory
plt.plot(state[:, 0], state[:, 2], label='Tracked Trajectory', color='blue', linewidth=2)

# Plot the desired trajectory based on the selected type
if args.trajectory == 'curvy':
    curvy_x = A * np.sin(omega * time)
    curvy_y = B * time
    plt.plot(curvy_x, curvy_y, '--', label='Desired Curvy Trajectory', color='red', linewidth=2)
else:
    circle_x = r * np.cos(omega * time)
    circle_y = r * np.sin(omega * time)
    plt.plot(circle_x, circle_y, '--', label='Desired Circular Trajectory', color='red', linewidth=2)

# Highlight the start and end points
plt.scatter(state[0, 0], state[0, 2], color='green', marker='o', s=100, label='Start Position')
plt.scatter(state[-1, 0], state[-1, 2], color='purple', marker='X', s=100, label='End Position')

# Label the axes
plt.xlabel('X Position [m]', fontsize=12)
plt.ylabel('Y Position [m]', fontsize=12)

# Set title
plt.title(f'4D {args.trajectory.capitalize()} Trajectory Tracking with Finite-Horizon LQG Controller', fontsize=14)

# Set the aspect ratio to be equal so the plot looks correct
plt.gca().set_aspect('equal', adjustable='box')

# Add a grid for better visibility
plt.grid(True, linestyle='--', alpha=0.7)

# Customize the legend position and style
plt.legend(loc='best', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()
