import numpy as np
from random import choices
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib.pyplot as plt

covariate = np.zeros((169))
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

def runMonte(k, trials = 1000):
    print(f"______RUN WITH K = {k}______")
    error_rates = []
    for _ in range(trials):
        N = 20**2 # total number of tiles assuming a tile size of 50x50

        fig, ax = plt.subplots(1, 1, figsize=(20,20))
        for i, img in range(enumerate()):
            plt.axvline(x=i, color='white')
            plt.axhline(y=i, color='white')

    n = 5
    test = np.arange(N)

    sampleIMGS = list(np.random.choice(np.arange(N), k, replace = True))
    sampled_image = [dectetorResult[x]["image_id"] for x in sampleIMGS]

    samplesTILES = choices(np.arange(N), k=n) # this will sample uniformly random n tiles with repetition
    f_s_i = [int(dectetorResult[x]["true_count"]) for x in samples]