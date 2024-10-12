#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def summarize_theta_w(J_DRCE_mean_all_samp, J_DRCE_std_all_samp, DRCE_prob_all_samp, theta_list, num_noise_list):
    fig = plt.figure(figsize=(6,4), dpi=300)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, num_noise in enumerate(num_noise_list):
        plt.plot(theta_list, J_DRCE_mean_all_samp[i], color=colors[i], marker='.', markersize=7, label=rf'$N={num_noise}$')
        plt.fill_between(theta_list, J_DRCE_mean_all_samp[i] + 0.25*J_DRCE_std_all_samp[i], J_DRCE_mean_all_samp[i] - 0.25*J_DRCE_std_all_samp[i], facecolor=colors[i], alpha=0.3)
        
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel(r'Out-Of-Sample Performance', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([theta_list[0], theta_list[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'/OSP.pdf', dpi=300, bbox_inches="tight")
    plt.clf()

    
    fig = plt.figure(figsize=(6,4), dpi=300)
    
    for i, num_noise in enumerate(num_noise_list):
        plt.plot(theta_list, DRCE_prob_all_samp[i], color=colors[i], marker='.', markersize=7, label=rf'$N={num_noise}$')
      
    plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel(r'Reliability', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xlim([theta_list[0], theta_list[-1]])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(path +'/OSP_Prob_{}_{}.pdf', dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', required=False, default="normal", type=str) #disurbance distribution (normal or uniform or quadratic)
    parser.add_argument('--noise_dist', required=False, default="normal", type=str) #noise distribution (normal or uniform or quadratic)
    args = parser.parse_args()
    
    theta_list = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] # radius of noise ambiguity set
    
    num_noise_list = [10, 15, 20] #
    
    noisedist = [args.noise_dist]
    
    J_DRCE_mean_all_samp = []
    J_DRCE_std_all_samp = []
    DRCE_prob_all_samp = []
            
    for noise_dist in noisedist:
        for idx, num_noise in enumerate(num_noise_list):
            J_DRCE_mean_samp = []
            J_DRCE_std_samp = []
            DRCE_prob_samp = []
            
            for theta in theta_list:
                path = "./results/{}_{}/finite/multiple/OS/".format(args.dist, args.noise_dist)

                theta_v_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                theta_w_ = f"_{str(theta).replace('.', '_')}" # change 1.0 to 1_0 for file name
                
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_mean_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_J_DRCE_mean = pickle.load(drce_file)
                drce_file.close()
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_std_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_J_DRCE_std = pickle.load(drce_file)
                drce_file.close()
                drce_file = open(path + 'N=' + str(num_noise) + '/drce_prob_os_' + theta_w_ + 'and' + theta_v_+ '.pkl', 'rb')
                output_DRCE_prob = pickle.load(drce_file)
                drce_file.close()
                
                
                J_DRCE_mean_samp.append(output_J_DRCE_mean[-1])
                J_DRCE_std_samp.append(output_J_DRCE_std[-1])
                
                DRCE_prob_samp.append(output_DRCE_prob[-1])
                
            J_DRCE_mean_all_samp.append(J_DRCE_mean_samp)
            J_DRCE_std_all_samp.append(J_DRCE_std_samp)
            DRCE_prob_all_samp.append(DRCE_prob_samp)                

    
    J_DRCE_mean_all_samp = np.array(J_DRCE_mean_all_samp)
    J_DRCE_std_all_samp = np.array(J_DRCE_std_all_samp)
    DRCE_prob_all_samp = np.array(DRCE_prob_all_samp)

    summarize_theta_w(J_DRCE_mean_all_samp, J_DRCE_std_all_samp, DRCE_prob_all_samp, theta_list, num_noise_list)

