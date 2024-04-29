import numpy as np
from random import choices
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib.pyplot as plt

dectetorResult = readFromcsv()

N = 169 #total number of images
true_total_strawberries = groundTruth()
F = true_total_strawberries

def get_tiles_from_image(im, tile_size=50):
    tiles = []
    for i in range(0, im.shape[0], tile_size):
        for j in range(0, im.shape[1], tile_size):
            tiles.append(im[i:i+tile_size, j:j+tile_size])
    return tiles


def monte_carlo(k, trials = 1000):
    error_rates = []
    print(f"______MONTE CARLO RUN WITH K = {k}______")
    for _ in range(trials):
        samples = list(np.random.choice(np.arange(N), k))
        f_s_i = [int(dectetorResult[x]["true_count"]) for x in samples] #pull the true count of strawberries
        F_hat = N * np.mean(f_s_i)
        err = np.abs(F-F_hat) / F
        error_rates.append(err * 100)

    mean_error = np.mean(error_rates)
    print(f"Mean error rate: {mean_error}%")
    return (k, mean_error)


k_coordinates = []
error_rate_coordiantes = []

set_of_k_values = [3, 5, 10, 12, 15, 17, 21, 25, 27, 30, 33, 38, 41, 45, 48, 51, 54, 57, 60]



for k in set_of_k_values:
    cur = monte_carlo(k)
    k_coordinates.append(cur[0])
    error_rate_coordiantes.append(cur[1])

fig, axes = plt.subplots(1, 1, figsize = (15, 15))
plt.plot(np.array(k_coordinates), np.array(error_rate_coordiantes))
plt.title("Relationship between error rate and number of samples verfied by human")
axes.set_xlabel("Number of samples verified by human")
axes.set_ylabel("Error rate (in percentage)")
axes.set_xticks([0, 5, 10, 15, 20])
plt.show()



