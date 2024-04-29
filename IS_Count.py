#choose intepreter of anaconda before running this

from matplotlib import pyplot as plt
from helper import readFromcsv, retreive_image, groundTruth
import skimage as ski
import numpy as np

#initialize covariate (red pixels count)
covariate = np.zeros((169)) 
image_data = readFromcsv()
image_strawberries_ids = [x["image_id"] for x in image_data]

#retrieve each image and convert them to hsv
img_arr = retreive_image(image_strawberries_ids)
hsv_img_arr = list(map(lambda rgb_img : ski.color.rgb2hsv(rgb_img), img_arr))
hue_img = list(map(lambda hsv_img : hsv_img[:,:,0], hsv_img_arr))

#count red pixels
for i, hue in enumerate(hue_img):
    cur = 0
    for row in hue:
        for pixel in row:
            if pixel > 0.95 or pixel < 0.04:
                cur += 1
    covariate[i] = int(cur)

#normalize our covariate
q= covariate / np.sum(covariate)

N = 169 #total number of images
true_total_strawberries = groundTruth()
F = true_total_strawberries

def run_is_count(k, trials = 1000):
    error_rates = []
    print(f"______ISCOUNT RUN WITH K = {k}______")
    for _ in range(trials):
        samples = list(np.random.choice(np.arange(N), k, p = q, replace = True))
        sampled_image = [image_strawberries_ids[x] for x in samples]
        
        f_s_i = [int(image_data[x]["true_count"]) for x in samples]

        #!Formula to calculate error rate, CI are copied from Guastavo's tutorial
        w_bar = 0
        for i, s_i in enumerate(samples):
            w_bar += f_s_i[i]/covariate.flatten()[s_i] # w_bar(S) = \sum(f/g)
        F_hat = np.sum(covariate)*w_bar/len(samples) # count estimate: F_hat = G(S)*1/n*w_bar(S)
        print('Estimated number of objects: %d (true count: %d)'%(F_hat, F))

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
    print(f"Mean Error rate for {k}:", mean_error)
    return (k, mean_error)

k_coordinates = []
error_rate_coordiantes = []

set_of_k_values = [3, 5, 10, 12, 15, 17, 21, 25, 27, 30, 33, 38, 41, 45, 48, 51, 54, 57, 60]



for k in set_of_k_values:
    cur = run_is_count(k)
    k_coordinates.append(cur[0])
    error_rate_coordiantes.append(cur[1])


fig, axes = plt.subplots(1, 1, figsize = (15, 15))
plt.plot(np.array(k_coordinates), np.array(error_rate_coordiantes))
plt.title("Relationship between error rate and number of samples verfied by human")
axes.set_xlabel("Number of samples verified by human")
axes.set_ylabel("Error rate (in percentage)")
axes.set_xticks([0, 5, 10, 15, 20])
plt.show()
