#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file generates Figure 5(a), (b)

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_noise(num_noise_list, avg_cost_lqg, std_cost_lqg, avg_cost_wdrc, std_cost_wdrc, avg_cost_drce, std_cost_drce, dist, noise_dist, infinite, path):

    t = np.array([5, 10, 15, 20, 25, 30, 35, 40])
    
    J_lqr_mean = np.array(avg_cost_lqg[0:])
    J_wdrc_mean = np.array(avg_cost_wdrc[0:])
    J_drce_mean = np.array(avg_cost_drce[0:])
    
    J_lqr_std = np.array(std_cost_lqg[0:])
    J_wdrc_std = np.array(std_cost_wdrc[0:])
    J_drce_std = np.array(std_cost_drce[0:])
    
    
    
    fig = plt.figure(figsize=(6,4), dpi=300)
    
    #----------------------------------------------
    plt.plot(t, J_lqr_mean, 'tab:red', label='LQG')
    plt.fill_between(t, J_lqr_mean + 0.25*J_lqr_std, J_lqr_mean - 0.25*J_lqr_std, facecolor='tab:red', alpha=0.3)
    
    plt.plot(t, J_wdrc_mean, 'tab:blue', label='WDRC [12]')
    plt.fill_between(t, J_wdrc_mean + 0.25*J_wdrc_std, J_wdrc_mean - 0.25*J_wdrc_std, facecolor='tab:blue', alpha=0.3)
    
    plt.plot(t, J_drce_mean, 'tab:green', label='WDR-CE [Ours]')
    plt.fill_between(t, J_drce_mean + 0.25*J_drce_std, J_drce_mean - 0.25*J_drce_std, facecolor='tab:green', alpha=0.3)
    
    plt.xlabel(r'Sample Size', fontsize=16)
    plt.ylabel(r'Total Cost', fontsize=16)
    plt.legend(fontsize=13)
    plt.grid()
    plt.xlim([t[0], t[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'/J_comp_{}_{}.pdf'.format(dist, noise_dist), dpi=300, bbox_inches="tight")
    plt.clf()
    print("Noise plot generated!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform)
    parser.add_argument('--application', required=False, action="store_true")
    parser.add_argument('--theta', required=False, default="0.1")
    parser.add_argument('--infinite', required=False, action="store_true") #infinite horizon settings if flagged
    args = parser.parse_args()
    
    
    print('\n-------Summary-------')
    if args.infinite:
        path = "./results/{}_{}/infinite/multiple/num_noise_plot".format(args.dist, args.noise_dist)
    else:
        path = "./results/{}_{}/finite/multiple/num_noise_plot".format(args.dist, args.noise_dist)
    num_noise_list = [5, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]
    avg_cost_lqg_file = open(path + '/lqg_mean.pkl', 'rb' )
    avg_cost_lqg = pickle.load(avg_cost_lqg_file)
    
    avg_cost_lqg_file.close()
    std_cost_lqg_file = open(path + '/lqg_std.pkl', 'rb' )
    std_cost_lqg = pickle.load(std_cost_lqg_file)
    std_cost_lqg_file.close()
    
    avg_cost_wdrc_file = open(path + '/wdrc_mean.pkl', 'rb' )
    avg_cost_wdrc = pickle.load(avg_cost_wdrc_file)
    #print(avg_cost_wdrc)
    avg_cost_wdrc_file.close()
    std_cost_wdrc_file = open(path + '/wdrc_std.pkl', 'rb' )
    std_cost_wdrc = pickle.load(std_cost_wdrc_file)
    std_cost_wdrc_file.close()
    
    avg_cost_drce_file = open(path + '/drce_mean.pkl', 'rb' )
    avg_cost_drce = pickle.load(avg_cost_drce_file)
    #print(avg_cost_drkf_wdrc)
    avg_cost_drce_file.close()
    std_cost_drce_file = open(path + '/drce_std.pkl', 'rb' )
    std_cost_drce = pickle.load(std_cost_drce_file)
    std_cost_drce_file.close()
    
    summarize_noise(num_noise_list, avg_cost_lqg, std_cost_lqg, avg_cost_wdrc, std_cost_wdrc, avg_cost_drce, std_cost_drce, args.dist, args.noise_dist, args.infinite ,path)
