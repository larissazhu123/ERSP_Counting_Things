from helper import readFromcsv, retreive_image
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
# i = 0
# for count in q:
#     i += count
# print(i) #close to 1, but not exactly

N = 169 #total number of images
human_verified_size = set({3, 5, 10, 15}) #choices of size of huma verfied samples

#Currently testing with k = 3 first
for k in human_verified_size:   
#return some index sample from the distribution
    samples = list(np.random.choice(np.arange(N), k, p = q, replace = True))
    sampled_image = [dectetorResult[x]["image_id"] for x in samples]
    numpy_images = retreive_image(sampled_image)
    fig, ax = plt.subplots(1, k, figsize = (20, 5 * k)) #create a Figure object with 3 slots
    for index, img in enumerate(numpy_images):
        plt.subplot(1,k,index + 1) #register a slot for the same Figure object, last parameter is current index of plot(start with 1)
        plt.imshow(img)
    plt.show()


