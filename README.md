Wasserstein Distributionally Robust Control and State Estimation for Partially Observable Linear Systems
====================================================

This repository includes the source code for implementing 
Linear-Quadratic-Gaussian(LQG), Wasserstein Distributionally Robust Controller(WDRC), Distributionally Robust Linear Quadratic Control (DRLQC), WDRC+DRKF, WDRC+DRMMSE,
and Distributionally Robust Control and Estimation(WDRCE) [Ours]

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)**
- (pickle5) if relevant error occurs
- joblib (>=1.4.2)
## Additional Requirements to run DRLQC
- Pytorch 2.0
- [Pymanopt] https://pymanopt.org/

## Code explanation
---
### Zero-mean Gaussian
First, generate the Total Cost data using
```
python main_param_zeromean.py
```
After data generation, plot the results using
```
python plot_params4_drlqc_zeromean.py
```
---
### Zero-mean U-Quadratic
First, generate the Total Cost data using
```
python main_param_zeromean.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_drlqc_zeromean.py --dist quadratic --noise_dist quadratic
```
---
### Nonzero-mean U-Quadratic
First, generate the Total Cost data using
```
python main_param_nonzeromean.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_drlqc_nonzeromean.py --dist quadratic --noise_dist quadratic
```
---
### Computation Time
First, generate the Computation_time data using
```
python main_time.py
```
Note that main_time.py is a time-consuming process.
After Data ge1neration, plot the results using
```
python plot_time.py
```
---
### Estimator Performance (a)
First, generate the Total Cost data using
```
python main_param_filter.py
```
After data generation, plot the results using
```
python plot_params4_F.py
```
---
### Estimator Performance (b)
First, generate the Total Cost data using
```
python main_param_filter.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params4_F.py --dist quadratic --noise_dist quadratic
```
---
### Long Horizon (a)
First, generate the Total Cost data using
```
python main_param_longT_parallel.py
```
After data generation, plot the results using
```
python plot_params_long.py
```
---
### Long Horizon (b)
First, generate the Total Cost data using
```
python main_param_longT_parallel.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_long.py --dist quadratic --noise_dist quadratic
```
---
### Out-of-sample Performacne (a), Reliability (b)
First, generate the data using
```
python main_OS_parallel.py
```
Be aware that this Out-of-Sample Experiment is a time consuming process.
After data generation, plot Figure (a), (b) using
```
python plot_osp.py
```
---
### Vehicle Control Problem : Noise sample size (a)
First, generate the Total Cost data using
```
python main_3.py
```
After data generation, plot the results using
```
python plot_J.py
```
---
### Vehicle Control Problem : Noise sample size (b)
First, generate the Total Cost data using
```
python main_3.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_J.py --dist quadratic --noise_dist quadratic
```
---
### Vehicle Control Problem : Total Cost (a)
First, generate the Total Cost data using
```
python main_param_s21.py
```
After data generation, plot the results using
```
python plot_params_21.py
```
---
### Vehicle Control Problem : Total Cost (b)
First, generate the Total Cost data using
```
python main_param_s21.py --dist quadratic --noise_dist quadratic
```
After data generation, plot the results using
```
python plot_params_21.py --dist quadratic --noise_dist quadratic
```
