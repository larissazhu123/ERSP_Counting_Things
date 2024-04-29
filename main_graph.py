import matplotlib.pyplot as plt
from new_disCount import run_dis_count
from IS_Count import run_is_count
from monte_carlo import monte_carlo
import numpy as np
import skimage as ski
from helper import groundTruth, readFromcsv, retreive_image


is_count_covariate = np.zeros((169))
discount_covariate = np.zeros((169))
N = 169
F = groundTruth()
image_data = readFromcsv()
image_strawberries_ids = [x["image_id"] for x in image_data]
#retreive image and convert to rgb
img_arr = retreive_image(image_strawberries_ids)

#initialize covariate (red pixels count)
hsv_img_arr = list(map(lambda rgb_img : ski.color.rgb2hsv(rgb_img), img_arr))
hue_img = list(map(lambda hsv_img : hsv_img[:,:,0], hsv_img_arr))

for i, hue in enumerate(hue_img):
    cur = 0
    for row in hue:
        for pixel in row:
            if pixel > 0.95 or pixel < 0.04:
                cur += 1
    is_count_covariate[i] = int(cur)

is_count_covariate = is_count_covariate / np.sum(is_count_covariate)

#initialize discount covariate (detector count)
for i, tempDict in enumerate(image_data):
    discount_covariate[i] = float(tempDict["approximate_count"])

discount_covariate = discount_covariate / np.sum(discount_covariate)


def run_3_method(k, trials = 1000):
    monte_carlo_error_rates = []
    is_count_error_rates = []
    discount_error_rates = []
    print(f"______RUN WITH K = {k}______")
    for _ in range(trials):
        #monte carlo
        samples_monte_carlo = list(np.random.choice(np.arange(N), k))
        f_s_i = [int(image_data[x]["true_count"]) for x in samples_monte_carlo]
        F_hat = N * np.mean(f_s_i)
        err = np.abs(F-F_hat) / F
        monte_carlo_error_rates.append(err * 100)


        #is count
        #!Formula copied from Guastavo's tutorial
        samples_is_count = list(np.random.choice(np.arange(N), k, p = is_count_covariate, replace = True))
        f_s_i = [int(image_data[x]["true_count"]) for x in samples_is_count]
        w_bar = 0
        for i, s_i in enumerate(samples_is_count):
            w_bar += f_s_i[i]/is_count_covariate.flatten()[s_i] 
        F_hat = np.sum(is_count_covariate)*w_bar/len(samples_is_count) 
        print('Estimated number of objects: %d (true count: %d)'%(F_hat, F))
        w_ci = 0
        for i, s_i in enumerate(samples_is_count):
            w_ci += (np.sum(is_count_covariate)*f_s_i[i]/is_count_covariate.flatten()[s_i] - F_hat)**2
        var_hat = w_ci/len(samples_is_count) # estimated variance
        CI = 1.96*np.sqrt(var_hat/len(samples_is_count)) # 95% confidence intervals
        err = np.abs(F-F_hat) / F
        is_count_error_rates.append(err * 100)

        #discount
        #!Formula copied from Guastavo's tutorial
        samples_discount = list(np.random.choice(np.arange(N), k, p = discount_covariate, replace = True))
        f_s_i = [int(image_data[x]["true_count"]) for x in samples_discount]
        w_bar = 0
        for i, s_i in enumerate(samples_discount):
            w_bar += f_s_i[i]/discount_covariate.flatten()[s_i] 
        F_hat = np.sum(discount_covariate)*w_bar/len(samples_discount) 
        print('Estimated number of objects: %d (true count: %d)'%(F_hat, F))
        w_ci = 0
        for i, s_i in enumerate(samples_discount):
            w_ci += (np.sum(discount_covariate)*f_s_i[i]/discount_covariate.flatten()[s_i] - F_hat)**2
        var_hat = w_ci/len(samples_discount) # estimated variance
        CI = 1.96*np.sqrt(var_hat/len(samples_discount)) # 95% confidence intervals
        err = np.abs(F-F_hat) / F
        discount_error_rates.append(err * 100)
    
    return (k, np.mean(monte_carlo_error_rates), np.mean(is_count_error_rates), np.mean(discount_error_rates))





k_vals = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
k_coordinates = []
monte_carlo_error_rates = []
is_count_error_rates = []
discount_error_rates = []
for k in k_vals:
    temp = run_3_method(k)
    k_coordinates.append(temp[0])
    monte_carlo_error_rates.append(temp[1])
    is_count_error_rates.append(temp[2])
    discount_error_rates.append(temp[3])

#plotting 
fig, axes = plt.subplots(1, 1, figsize = (15, 15))
axes.plot(np.array(k_coordinates), np.array(monte_carlo_error_rates), label = "Monte Carlo")
axes.plot(np.array(k_coordinates), np.array(is_count_error_rates), label = "IS Count")
axes.plot(np.array(k_coordinates), np.array(discount_error_rates), label = "Discount")
    






