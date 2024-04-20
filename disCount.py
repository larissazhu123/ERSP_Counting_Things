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
human_verified_size = set({3, 5, 10}) #choices of size of huma verfied samples
true_total_strawberries = groundTruth()
F = true_total_strawberries

# for k in human_verified_size:   
#return some index sample from the distribution
k = 3
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

#TODO: get input of count for each image
f_s_i = []

w_bar = 0
for i, s_i in enumerate(samples):
  w_bar += f_s_i[i]/covariate.flatten()[s_i] # w_bar(S) = \sum(f/g)
F_hat = np.sum(covariate)*w_bar/len(samples) # count estimate: F_hat = G(S)*1/n*w_bar(S)
print('Estimated number of objects: %d (true count: %d)'%(F_hat, F))

