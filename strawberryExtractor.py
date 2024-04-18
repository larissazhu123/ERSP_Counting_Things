# we know: range of file's ID
# folder: images_384_VarV2
# goal: extract file names of strawberries

import os

# strawberries_file_range = [285, 351]

strawberries = []

with open("./data/ImageClasses_FSC147.txt", "r") as f:
    output = f.readlines()
    # iterate through every line
    for line in output:
        # format line
        filename, img_cls = line.split('\t')
        img_cls = img_cls.replace('\n','')

        if img_cls == "strawberries":
            strawberries.append(filename)

print(len(strawberries))

with open("./extractedStrawberries.txt", "w+") as f:
    for straw in strawberries:
        f.write(straw + "\n") 


# run each strawberry file, get:
# annotation from annotation_FSC147_384.json (true count)
# approximation count by running test.py 

# with open("./strawberries.csv", "r", newline="") as f:
































