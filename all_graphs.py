from helper import readFromcsv, retreive_image, groundTruth
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data
from skimage.color import rgb2hsv
from matplotlib import pyplot as plt
from helper import readFromcsv, retreive_image, groundTruth
import skimage as ski
import numpy as np

from new_disCount import run_dis_count
from monte_carlo import get_tiles_from_image, monte_carlo
from IS_Count import retreive_image, run_is_count

run_dis_count()
monte_carlo()
run_is_count()


k_coordinates = []
error_rate_coordiantes = []

set_of_k_values = [3, 5, 10, 12, 15, 17, 21, 25, 27, 30, 33, 38, 41, 45, 48, 51, 54, 57, 60]



for k in set_of_k_values:
    cur = monte_carlo(k)
    k_coordinates.append(cur[0])
    error_rate_coordiantes.append(cur[1])



