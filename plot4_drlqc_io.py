#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from matplotlib import cm
from scipy.interpolate import interp1d

def summarize(out_lq_list, out_wdrc_list, out_drce_list, out_drlqc_list, dist, noise_dist, path, num,  plot_results=True):
    x_lqr_list, J_lqr_list, y_lqr_list, u_lqr_list = [], [], [], []
    x_wdrc_list, J_wdrc_list, y_wdrc_list, u_wdrc_list = [], [], [], [] # original wdrc with ordinary Kalman Filter
    x_drce_list, J_drce_list, y_drce_list, u_drce_list = [], [], [], [] # drce
    x_drlqc_list, J_drlqc_list, y_drlqc_list, u_drlqc_list = [], [], [], [] # drce
    time_wdrc_list, time_lqr_list, time_drce_list, time_drlqc_list = [], [], [], []


    for out in out_lq_list:
         x_lqr_list.append(out['state_traj'])
         J_lqr_list.append(out['cost'])
         y_lqr_list.append(out['output_traj'])
         u_lqr_list.append(out['control_traj'])
         time_lqr_list.append(out['comp_time'])
         
    x_lqr_mean, J_lqr_mean, y_lqr_mean, u_lqr_mean = np.mean(x_lqr_list, axis=0), np.mean(J_lqr_list, axis=0), np.mean(y_lqr_list, axis=0), np.mean(u_lqr_list, axis=0)
    x_lqr_std, J_lqr_std, y_lqr_std, u_lqr_std = np.std(x_lqr_list, axis=0), np.std(J_lqr_list, axis=0), np.std(y_lqr_list, axis=0), np.std(u_lqr_list, axis=0)
    time_lqr_ar = np.array(time_lqr_list)
    print("LQG cost : ", J_lqr_mean[0])
    print("LQG cost std : ", J_lqr_std[0])
    J_lqr_ar = np.array(J_lqr_list)
    
    
    for out in out_wdrc_list:
        x_wdrc_list.append(out['state_traj'])
        J_wdrc_list.append(out['cost'])
        y_wdrc_list.append(out['output_traj'])
        u_wdrc_list.append(out['control_traj'])
        time_wdrc_list.append(out['comp_time'])
    x_wdrc_mean, J_wdrc_mean, y_wdrc_mean, u_wdrc_mean = np.mean(x_wdrc_list, axis=0), np.mean(J_wdrc_list, axis=0), np.mean(y_wdrc_list, axis=0), np.mean(u_wdrc_list, axis=0)
    x_wdrc_std, J_wdrc_std, y_wdrc_std, u_wdrc_std = np.std(x_wdrc_list, axis=0), np.std(J_wdrc_list, axis=0), np.std(y_wdrc_list, axis=0), np.std(u_wdrc_list, axis=0)
    time_wdrc_ar = np.array(time_wdrc_list)
    print("WDRC cost : ", J_wdrc_mean[0])
    print("WDRC cost std : ", J_wdrc_std[0])
    J_wdrc_ar = np.array(J_wdrc_list)



    for out in out_drce_list:
            x_drce_list.append(out['state_traj'])
            J_drce_list.append(out['cost'])
            y_drce_list.append(out['output_traj'])
            u_drce_list.append(out['control_traj'])
            time_drce_list.append(out['comp_time'])
    x_drce_mean, J_drce_mean, y_drce_mean, u_drce_mean = np.mean(x_drce_list, axis=0), np.mean(J_drce_list, axis=0), np.mean(y_drce_list, axis=0), np.mean(u_drce_list, axis=0)
    x_drce_std, J_drce_std, y_drce_std, u_drce_std = np.std(x_drce_list, axis=0), np.std(J_drce_list, axis=0), np.std(y_drce_list, axis=0), np.std(u_drce_list, axis=0)
    time_drce_ar = np.array(time_drce_list)
    print("DRCE cost : ", J_drce_mean[0])
    print("DRCE cost std : ", J_drce_std[0])
    J_drce_ar = np.array(J_drce_list)   
    nx = x_drce_mean.shape[1]
    T = u_drce_mean.shape[0]
    
    for out in out_drlqc_list:
            x_drlqc_list.append(out['state_traj'])
            J_drlqc_list.append(out['cost'])
            y_drlqc_list.append(out['output_traj'])
            u_drlqc_list.append(out['control_traj'])
            time_drlqc_list.append(out['comp_time'])
    x_drlqc_mean, J_drlqc_mean, y_drlqc_mean, u_drlqc_mean = np.mean(x_drlqc_list, axis=0), np.mean(J_drlqc_list, axis=0), np.mean(y_drlqc_list, axis=0), np.mean(u_drlqc_list, axis=0)
    x_drlqc_std, J_drlqc_std, y_drlqc_std, u_drlqc_std = np.std(x_drlqc_list, axis=0), np.std(J_drlqc_list, axis=0), np.std(y_drlqc_list, axis=0), np.std(u_drlqc_list, axis=0)
    time_drlqc_ar = np.array(time_drlqc_list)
    print("DRLQC cost : ", J_drlqc_mean[0])
    print("DRLQC cost std : ", J_drlqc_std[0])
    J_drlqc_ar = np.array(J_drlqc_list)   
    nx = x_drlqc_mean.shape[1]
    T = u_drlqc_mean.shape[0]
    
    
    
    # ------------------------------------------------------------
    if plot_results:
        nx = x_drce_mean.shape[1]
        T = u_drce_mean.shape[0]
        nu = u_drce_mean.shape[1]
        ny= y_drce_mean.shape[1]

        fig = plt.figure(figsize=(6,4), dpi=300)

        t = np.arange(T+1)
        for i in range(nx):

            if x_lqr_list != []:
                plt.plot(t, x_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, x_lqr_mean[:,i, 0] + 0.3*x_lqr_std[:,i,0],
                               x_lqr_mean[:,i,0] - 0.3*x_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if x_wdrc_list != []:
                plt.plot(t, x_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, x_wdrc_mean[:,i,0] + 0.3*x_wdrc_std[:,i,0],
                                x_wdrc_mean[:,i,0] - 0.3*x_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if x_drlqc_list != []:
                plt.plot(t, x_drlqc_mean[:,i,0], 'tab:purple', label='DRLQC')
                plt.fill_between(t, x_drlqc_mean[:,i, 0] + 0.3*x_drlqc_std[:,i,0],
                               x_drlqc_mean[:,i,0] - 0.3*x_drlqc_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            if x_drce_list != []:
                plt.plot(t, x_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, x_drce_mean[:,i, 0] + 0.3*x_drce_std[:,i,0],
                               x_drce_mean[:,i,0] - 0.3*x_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
                
            plt.xlabel(r'$t$', fontsize=22)
            plt.ylabel(r'$x_{{{}}}$'.format(i+1), fontsize=22)
            plt.legend(fontsize=20)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'states_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T)
        for i in range(nu):

            if u_lqr_list != []:
                plt.plot(t, u_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, u_lqr_mean[:,i,0] + 0.25*u_lqr_std[:,i,0],
                             u_lqr_mean[:,i,0] - 0.25*u_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if u_wdrc_list != []:
                plt.plot(t, u_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, u_wdrc_mean[:,i,0] + 0.25*u_wdrc_std[:,i,0],
                                u_wdrc_mean[:,i,0] - 0.25*u_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if u_drlqc_list != []:
                plt.plot(t, u_drlqc_mean[:,i,0], 'tab:purple', label='DRLQC')
                plt.fill_between(t, u_drlqc_mean[:,i,0] + 0.25*u_drlqc_std[:,i,0],
                             u_drlqc_mean[:,i,0] - 0.25*u_drlqc_std[:,i,0], facecolor='tab:purple', alpha=0.3) 
            if u_drce_list != []:
                plt.plot(t, u_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, u_drce_mean[:,i,0] + 0.25*u_drce_std[:,i,0],
                             u_drce_mean[:,i,0] - 0.25*u_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)       
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$u_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'controls_{}_{}_{}_{}.pdf'.format(i+1, num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()

        t = np.arange(T+1)
        for i in range(ny):
            if y_lqr_list != []:
                plt.plot(t, y_lqr_mean[:,i,0], 'tab:red', label='LQG')
                plt.fill_between(t, y_lqr_mean[:,i,0] + 0.25*y_lqr_std[:,i,0],
                             y_lqr_mean[:,i, 0] - 0.25*y_lqr_std[:,i,0], facecolor='tab:red', alpha=0.3)
            if y_wdrc_list != []:
                plt.plot(t, y_wdrc_mean[:,i,0], 'tab:blue', label='WDRC')
                plt.fill_between(t, y_wdrc_mean[:,i,0] + 0.25*y_wdrc_std[:,i,0],
                                y_wdrc_mean[:,i, 0] - 0.25*y_wdrc_std[:,i,0], facecolor='tab:blue', alpha=0.3)
            if y_drlqc_list != []:
                plt.plot(t, y_drlqc_mean[:,i,0], 'tab:purple', label='DRLQC')
                plt.fill_between(t, y_drlqc_mean[:,i,0] + 0.25*y_drlqc_std[:,i,0],
                             y_drlqc_mean[:,i, 0] - 0.25*y_drlqc_std[:,i,0], facecolor='tab:purple', alpha=0.3)
            if y_drce_list != []:
                plt.plot(t, y_drce_mean[:,i,0], 'tab:green', label='WDR-CE')
                plt.fill_between(t, y_drce_mean[:,i,0] + 0.25*y_drce_std[:,i,0],
                             y_drce_mean[:,i, 0] - 0.25*y_drce_std[:,i,0], facecolor='tab:green', alpha=0.3)
            
            plt.xlabel(r'$t$', fontsize=16)
            plt.ylabel(r'$y_{{{}}}$'.format(i+1), fontsize=16)
            plt.legend(fontsize=16)
            plt.grid()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim([t[0], t[-1]])
            ax = fig.gca()
            ax.locator_params(axis='y', nbins=5)
            ax.locator_params(axis='x', nbins=5)
            fig.set_size_inches(6, 4)
            plt.savefig(path +'outputs_{}_{}_{}_{}.pdf'.format(i+1,num, dist, noise_dist), dpi=300, bbox_inches="tight")
            plt.clf()


        plt.title('Optimal Value')
        t = np.arange(T+1)

        if J_lqr_list != []:
            plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
            plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
        if J_wdrc_list != []:
            plt.plot(t, J_wdrc_mean, 'tab:blue', label='WDRC')
            plt.fill_between(t, J_wdrc_mean + 0.25*J_wdrc_std, J_wdrc_mean - 0.25*J_wdrc_std, facecolor='tab:blue', alpha=0.3)
        if J_drlqc_list != []:
            plt.plot(t, J_drlqc_mean, 'tab:purple', label='DRLQC')
            plt.fill_between(t, J_drlqc_mean + 0.25*J_drlqc_std, J_drlqc_mean - 0.25*J_drlqc_std, facecolor='tab:purple', alpha=0.3)
        if J_drce_list != []:
            plt.plot(t, J_drce_mean, 'tab:green', label='WDR-CE')
            plt.fill_between(t, J_drce_mean + 0.25*J_drce_std, J_drce_mean - 0.25*J_drce_std, facecolor='tab:green', alpha=0.3)
        
        plt.xlabel(r'$t$', fontsize=16)
        plt.ylabel(r'$V_t(x_t)$', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.xlim([t[0], t[-1]])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        plt.clf()


        ax = fig.gca()
        t = np.arange(T+1)
        
        max_bin = np.max([J_wdrc_ar[:,0], J_lqr_ar[:,0], J_drce_ar[:,0], J_drlqc_ar[:,0]])
        min_bin = np.min([J_wdrc_ar[:,0], J_lqr_ar[:,0], J_drce_ar[:,0], J_drlqc_ar[:,0]])


        
        ax.hist(J_lqr_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:red', label='LQG', alpha=0.5, linewidth=0.5, edgecolor='tab:red')
        ax.hist(J_wdrc_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:blue', label='WDRC', alpha=0.5, linewidth=0.5, edgecolor='tab:blue')
        ax.hist(J_drlqc_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:purple', label='DRLQC', alpha=0.5, linewidth=0.5, edgecolor='tab:purple')
        ax.hist(J_drce_ar[:,0], bins=50, range=(min_bin,max_bin), color='tab:green', label='WDR-CE', alpha=0.5, linewidth=0.5, edgecolor='tab:green')
        
        ax.axvline(J_wdrc_ar[:,0].mean(), color='navy', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_lqr_ar[:,0].mean(), color='maroon', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_drlqc_ar[:,0].mean(), color='purple', linestyle='dashed', linewidth=1.5)
        ax.axvline(J_drce_ar[:,0].mean(), color='green', linestyle='dashed', linewidth=1.5)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        handles, labels = plt.gca().get_legend_handles_labels()
        
        order = [0, 1, 2, 3]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

        ax.grid()
        ax.set_axisbelow(True)
        #plt.title('{} system disturbance, {} observation noise'.format(dist, noise_dist))
        plt.xlabel(r'Total Cost', fontsize=16)
        plt.ylabel(r'Frequency', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(path +'J_hist_{}_{}_{}.pdf'.format(num, dist, noise_dist), dpi=300, bbox_inches="tight")
        plt.clf()


        plt.close('all')
        
        
    print( 'cost_lqr:{} ({})'.format(J_lqr_mean[0],J_lqr_std[0]),'cost_WDRC: {} ({})'.format(J_wdrc_mean[0], J_wdrc_std[0]) , 'cost_wdrce:{} ({})'.format(J_drce_mean[0],J_drce_std[0]), 'cost_wdrlqc:{} ({})'.format(J_drlqc_mean[0],J_drlqc_std[0]))
    print( 'time_lqr: {} ({})'.format(time_lqr_ar.mean(), time_lqr_ar.std()),'time_WDRC: {} ({})'.format(time_wdrc_ar.mean(), time_wdrc_ar.std()), 'time_wdrce: {} ({})'.format(time_drce_ar.mean(), time_drce_ar.std()), 'time_wdrlqc: {} ({})'.format(time_drlqc_ar.mean(), time_drlqc_ar.std()))
    #print( 'Settling time_lqr: {}'.format(SettlingTime_lqr),'Settling time_WDRC: {} '.format(SettlingTime), 'Settling time_drce: {}'.format(SettlingTime_drce))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    parser.add_argument('--num_sim', required=False, default=500, type=int)
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    parser.add_argument('--use_lambda', required=False, action="store_true") #use lambda results if flagged
    args = parser.parse_args()

    horizon = "finite"
        
    if args.use_lambda:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/io/".format(args.dist, args.noise_dist)
        rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_lambda/io/raw/".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/io/".format(args.dist, args.noise_dist)
        rawpath = "./results/{}_{}/finite/multiple/DRLQC/params_thetas/io/raw/".format(args.dist, args.noise_dist)

    #Load data
    drlqc_theta_w_values =[]
    drlqc_lambda_values = []
    drlqc_theta_v_values = []
    drlqc_cost_values = []
    
    drce_theta_w_values =[]
    drce_lambda_values = []
    drce_theta_v_values = []
    drce_cost_values = []
    
    wdrc_theta_w_values = []
    wdrc_lambda_values = []
    wdrc_theta_v_values = []
    wdrc_cost_values = []
    
    lqg_theta_w_values =[]
    lqg_lambda_values = []
    lqg_theta_v_values = []
    lqg_cost_values = []
    
    drlqc_optimal_theta_w, drlqc_optimal_theta_v, drlqc_optimal_cost = 0, 0, 99999999
    drce_optimal_theta_w, drce_optimal_theta_v, drce_optimal_cost = 0, 0, 99999999
    wdrc_optimal_theta_w, wdrc_optimal_cost = 0, 99999999
    
    
    # TODO : Modify the theta_v_list and lambda_list below to match your experiments!!! 
    
    if args.dist=='normal':
        lambda_list = [12, 15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter
        theta_v_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        theta_w_list = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    else:
        lambda_list = [15, 20, 25, 30, 35, 40, 45, 50] # disturbance distribution penalty parameter
        theta_v_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        theta_w_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        
    
    # Regular expression pattern to extract numbers from file names
    
    if args.use_lambda:
        pattern_drce = r"drce_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drlqc = r"drlqc_(\d+)and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_(\d+)"
    else:
        pattern_drlqc = r"drlqc_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_drce = r"drce_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?and_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
        pattern_wdrc = r"wdrc_(\d+(?:\.\d+)?)_?(\d+(?:_\d+)?)?"
    pattern_lqg = r"lqg.pkl"
    
    print(path)
    # Iterate over each file in the directory
    for filename in os.listdir(path):
        match = re.search(pattern_drce, filename)
        if match:
            if args.use_lambda:
                lambda_value = float(match.group(1))  # Extract lambda
                theta_v_value = float(match.group(2))   # Extract theta_v value
                theta_v_str = match.group(3)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_lambda_values.append(lambda_value)
                drce_theta_v_values.append(theta_v_value)
            else:
                theta_w_value = float(match.group(1))  # Extract theta_w value
                theta_w_str = match.group(2)
                theta_w_value += float(theta_w_str)/10
                theta_v_value = float(match.group(3))   # Extract theta_v value
                theta_v_str = match.group(4)
                theta_v_value += float(theta_v_str)/10
                #changed _1_5_ to 1.5!
                # Store theta_w and theta values
                drce_theta_w_values.append(theta_w_value)
                drce_theta_v_values.append(theta_v_value)
            
            drce_file = open(path + filename, 'rb')
            drce_cost = pickle.load(drce_file)
            if drce_cost[0]<drce_optimal_cost:
                drce_optimal_cost = drce_cost[0]
                drce_optimal_theta_w = theta_w_value
                drce_optimal_theta_v = theta_v_value
            drce_file.close()
            drce_cost_values.append(drce_cost[0])  # Store cost value
        else:
            match_drlqc = re.search(pattern_drlqc, filename)
            if match_drlqc:
                if args.use_lambda:
                    lambda_value = float(match_drlqc.group(1))  # Extract lambda
                    theta_v_value = float(match_drlqc.group(2))   # Extract theta_v value
                    theta_v_str = match_drlqc.group(3)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drlqc_lambda_values.append(lambda_value)
                    drlqc_theta_v_values.append(theta_v_value)
                else:
                    theta_w_value = float(match_drlqc.group(1))  # Extract theta_w value
                    theta_w_str = match_drlqc.group(2)
                    theta_w_value += float(theta_w_str)/10
                    theta_v_value = float(match_drlqc.group(3))   # Extract theta_v value
                    theta_v_str = match_drlqc.group(4)
                    theta_v_value += float(theta_v_str)/10
                    #changed _1_5_ to 1.5!
                    # Store theta_w and theta values
                    drlqc_theta_w_values.append(theta_w_value)
                    drlqc_theta_v_values.append(theta_v_value)
                
                drlqc_file = open(path + filename, 'rb')
                drlqc_cost = pickle.load(drlqc_file)
                if drlqc_cost[0]<drlqc_optimal_cost:
                    drlqc_optimal_cost = drlqc_cost[0]
                    drlqc_optimal_theta_w = theta_w_value
                    drlqc_optimal_theta_v = theta_v_value
                drlqc_file.close()
                drlqc_cost_values.append(drlqc_cost[0])  # Store cost value
            else:
                match_wdrc = re.search(pattern_wdrc, filename)
                if match_wdrc: # wdrc
                    if args.use_lambda:
                        lambda_value = float(match_wdrc.group(1))  # Extract lambda
                    else:
                        theta_w_value = float(match_wdrc.group(1))  # Extract theta_w value
                        theta_w_str = match_wdrc.group(2)
                        theta_w_value += float(theta_w_str)/10
                    wdrc_file = open(path + filename, 'rb')
                    wdrc_cost = pickle.load(wdrc_file)
                    if wdrc_cost[0] < wdrc_optimal_cost:
                        #print("WDRC!!!{} & {}".format(wdrc_cost[0], theta_w_value))
                        wdrc_optimal_cost = wdrc_cost[0]
                        wdrc_optimal_theta_w = theta_w_value
                    wdrc_file.close()
                    for aux_theta_v in theta_v_list:
                        if args.use_lambda:
                            wdrc_lambda_values.append(lambda_value)
                        else:
                            wdrc_theta_w_values.append(theta_w_value)
                        wdrc_theta_v_values.append(aux_theta_v) # since wdrc not affected by theta v, just add auxilary theta for plot
                        wdrc_cost_values.append(wdrc_cost[0])
                else:
                    match_lqg = re.search(pattern_lqg, filename)
                    if match_lqg:
                        lqg_file = open(path + filename, 'rb')
                        lqg_cost = pickle.load(lqg_file)
                        
                        lqg_file.close()
                        if args.use_lambda:
                            for aux_lambda in lambda_list:
                                for aux_theta_v in theta_v_list:
                                    lqg_lambda_values.append(aux_lambda)
                                    lqg_theta_v_values.append(aux_theta_v)
                                    lqg_cost_values.append(lqg_cost[0])
                        else:
                            for aux_theta_w in theta_w_list:
                                for aux_theta_v in theta_v_list:
                                    lqg_theta_w_values.append(aux_theta_w)
                                    lqg_theta_v_values.append(aux_theta_v)
                                    lqg_cost_values.append(lqg_cost[0])
                
                    
    # We obtained the best-parameters for each method (within the examined region)
    # DRLQC
    print("Best parameters & Cost within the examined region")
    print("-------------------------")
    print("DRLQC")
    print("Best theta_w: {}, Best theta_v: {}, Best cost: {}".format(drlqc_optimal_theta_w, drlqc_optimal_theta_v, drlqc_optimal_cost))
    print("-------------------------")
    print("DRCE")
    print("Best theta_w: {}, Best theta_v: {}, Best cost: {}".format(drce_optimal_theta_w, drce_optimal_theta_v, drce_optimal_cost))
    print("-------------------------")
    print("WDRC")
    print("Best theta_w: {},  Best cost: {}".format(wdrc_optimal_theta_w, wdrc_optimal_cost))
    print("-------------------------")
    print("LQG")
    print("Cost: {}".format(lqg_cost[0]))
    
    #REMOVE BELOW
    # drlqc_optimal_theta_w = 3.0
    # drlqc_optimal_theta_v = 10.0
    # drce_optimal_theta_w = 2.0
    # drce_optimal_theta_v = 10.0
    # wdrc_optimal_theta_w = 2.0
    
    # Now, pass the raw data for each methods using optimal parameters
    drlqc_theta_w_str = str(drlqc_optimal_theta_w).replace('.', '_')
    drlqc_theta_v_str = str(drlqc_optimal_theta_v).replace('.', '_')

    drce_theta_w_str = str(drce_optimal_theta_w).replace('.', '_')
    drce_theta_v_str = str(drce_optimal_theta_v).replace('.', '_')
    
    wdrc_theta_w_str = str(wdrc_optimal_theta_w).replace('.', '_')
    
    # Construct the filename for the optimal parameters
    drlqc_filename = f"drlqc_{drlqc_theta_w_str}and_{drlqc_theta_v_str}.pkl"
    drlqc_filepath = rawpath + drlqc_filename
    
    drce_filename = f"drce_{drce_theta_w_str}and_{drce_theta_v_str}.pkl"
    drce_filepath = rawpath + drce_filename
    
    wdrc_filename = f"wdrc_{wdrc_theta_w_str}.pkl"
    wdrc_filepath = rawpath + wdrc_filename
    
    lqg_filename = f"lqg.pkl"
    lqg_filepath = rawpath + lqg_filename

    # Load the data from the file
    with open(drlqc_filepath, 'rb') as drlqc_file:
        drlqc_data = pickle.load(drlqc_file)
    with open(drce_filepath, 'rb') as drce_file:
        drce_data = pickle.load(drce_file)
    with open(wdrc_filepath, 'rb') as wdrc_file:
        wdrc_data = pickle.load(wdrc_file)
    with open(lqg_filepath, 'rb') as lqg_file:
        lqg_data = pickle.load(lqg_file)

    
    print("Loaded data for with optimal parameters: DRLQC")
    #print(drlqc_data['cost'])


    summarize(lqg_data, wdrc_data, drce_data, drlqc_data, args.dist, args.noise_dist,  path , args.num_sim, plot_results=True)
    

