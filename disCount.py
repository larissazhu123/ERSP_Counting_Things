from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib.pyplot as plt
#! Note: since this is a just a count (a number), not a tile like in Guastavo's tutorial,
#! I use a single vector to represent our covariate, not sure if this is correct yet 
#initialize covariate (detector count)
covariate = np.zeros((169))
dectetorResult = readFromcsv()
for i, tempDict in enumerate(dectetorResult):
    covariate[i] = float(tempDict["approximate_count"])

#normalize our covariate
q = covariate / np.sum(covariate)


N = 169 #total number of images
true_total_strawberries = groundTruth()
F = true_total_strawberries

def run_dis_count(k):
    samples = list(np.random.choice(np.arange(N), k, p = q, replace = True))
    sampled_image = [dectetorResult[x]["image_id"] for x in samples]
    numpy_images = retreive_image(sampled_image)
    fig, ax = plt.subplots(1, k, figsize = (20, 10)) #create a Figure object with k slots, size is 20 inches width, 10 inches height
    curIndex = 0
    for index, img in enumerate(numpy_images):
        ax[curIndex].imshow(img)
        ax[curIndex].axis("off")
        ax[curIndex].set_title(f"Image {curIndex}")
        curIndex += 1
    plt.show()

    #here instead of us manually counting, we already got the true count, just retrieve it directly and put into the array
    f_s_i = [int(dectetorResult[x]["true_count"]) for x in samples]

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

# And here we calculate the error
Error = np.abs(F - F_hat)/F
print('Error rate: %.2f%%'%(Error*100.))
print('Estimate: %d \u00B1 %d'%(F_hat, CI))

#TODO: essentially do the same thing for different k values, record pair (k, error_rate) to construct the graph between k and error rate