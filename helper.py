
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#return a list of DictRead object where each object will have the following field: image_id, approximate_count, true_count
def readFromcsv() -> list[dict]:
    with open("./strawberries.csv", "r") as file:
        content = csv.DictReader(file)
        return list(content) 
    

#return list of numpy array where each array is asssociated to an image included in the input list
def retreive_image(list_of_image_ids : list[str]):
    res = []
    for id in list_of_image_ids:
        path = "./data/images_384_VarV2/" + id
        curImg = Image.open(path)
        res.append(np.array(curImg))
    return res

#if run this file directly
if (__name__ == "__main__"):
    # print(readFromcsv())
    test_np_array = retreive_image(["285.jpg"])
