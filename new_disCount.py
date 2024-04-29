"""""
This file is for the new disCount, in which for each k we would want to run 100 trials and record the mean error
"""

from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data
from skimage.color import rgb2hsv

#initialize covariate (detector count)

#TODO: modify this code so that each k will be running with 100 trials, return (k, mean_of_all_error_trials)
def run_dis_count(k, trials = 1000):
    covariate = np.zeros((169))
    dectetorResult = readFromcsv()
    for i, tempDict in enumerate(dectetorResult):
        covariate[i] = float(tempDict["approximate_count"])
    #normalize our covariate
    q= covariate / np.sum(covariate)
    N = 169 #total number of images
    true_total_strawberries = groundTruth()
    F = true_total_strawberries



    error_rates = []
    # print(f"______DISCOUNT RUN WITH K = {k}______")
    for _ in range(trials):
        samples = list(np.random.choice(np.arange(N), k, p = q, replace = True))
        
        f_s_i = [int(dectetorResult[x]["true_count"]) for x in samples]

        #!Formula to calculate error rate, CI are copied from Guastavo's tutorial
        w_bar = 0
        for i, s_i in enumerate(samples):
            w_bar += f_s_i[i]/covariate.flatten()[s_i] # w_bar(S) = \sum(f/g)
        F_hat = np.sum(covariate)*w_bar/len(samples) # count estimate: F_hat = G(S)*1/n*w_bar(S)
        # print('Estimated number of objects: %d (true count: %d)'%(F_hat, F))

        # Calculating confidence intervals
        w_ci = 0
        for i, s_i in enumerate(samples):
            w_ci += (np.sum(covariate)*f_s_i[i]/covariate.flatten()[s_i] - F_hat)**2
        var_hat = w_ci/len(samples) # estimated variance
        CI = 1.96*np.sqrt(var_hat/len(samples)) # 95% confidence intervals

        err = np.abs(F-F_hat) / F
        error_rates.append(err * 100)

    # let's get the mean error
    mean_error = np.mean(error_rates)
    # print(f"Mean Error rate for {k}:", mean_error)
    return (k, mean_error)
    
    # And here we calculate the error
    # Error = np.abs(F - F_hat)/F
    # print('Error rate: %.2f%%'%(Error*100.))
    # print('Estimate: %d \u00B1 %d'%(F_hat, CI))
    # return (k , Error * 100)


k_coordinates = []
error_rate_coordiantes = []

set_of_k_values = [3, 5, 10, 12, 15, 17, 21, 25, 27, 30, 33, 38, 41, 45, 48, 51, 54, 57, 60]



for k in set_of_k_values:
    cur = run_dis_count(k)
    k_coordinates.append(cur[0])
    error_rate_coordiantes.append(cur[1])







    



# fig, axes = plt.subplots(1, 1, figsize = (15, 15))
# plt.plot(np.array(k_coordinates), np.array(error_rate_coordiantes))
# plt.title("Relationship between error rate and number of samples verfied by human")
# axes.set_xlabel("Number of samples verified by human")
# axes.set_ylabel("Error rate (in percentage)")
# axes.set_xticks([0, 5, 10, 15, 20])
# plt.show()
