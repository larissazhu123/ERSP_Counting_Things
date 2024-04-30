from helper import readFromcsv, retreive_image 
#from helper import groundTruth
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import skimage as ski
from skimage import data
from skimage.color import rgb2hsv
import cv2
import functools

def groundTruth() -> int:
    tempList = readFromcsv()
    return functools.reduce(lambda a, b: a + b, [float(x["true_count"]) for x in tempList])

imgs = retreive_image([x["image_id"] for x in readFromcsv()])


def hsv_cov():
    for img in imgs[:1]:
        #idea one (but it looks more green than anything doesn't rlly work)
        image = img 
        img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([70,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([120,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join masks
        mask = mask0+mask1

        # set output img to zero everywhere except mask
        output_img = img.copy()
        output_img[np.where(mask==0)] = 0

        output_hsv = img_hsv.copy()
        output_hsv[np.where(mask==0)] = 0

        plt.imshow(output_hsv)
        plt.show()
        

        #idea two just blacking out everything except for red (doesn't work)
        """image = img
        img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,175,20])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,75,20])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0 + mask1
        new_img = cv2.bitwise_and(image,image, mask = mask)
        plt.imshow(new_img)
        plt.show()"""


hsv_cov()

