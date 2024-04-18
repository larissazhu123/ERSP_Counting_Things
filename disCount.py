from readDetectorCount import readFromcsv
import numpy as np

#! Note: since this is a just a count (a number), not a tile like in Guastavo's tutorial,
#! I use a single vector to represent our covariate, not sure if this is correct yet 
#initialize covariate (detector count)
covariate = np.zeros((169))
dectetorCount = readFromcsv()
for i, count in enumerate(dectetorCount[1:]):
    covariate[i] = float(count)

#normalize our covariate
q = covariate / np.sum(covariate)
i = 0

for count in q:
    i += count
print(i) #close to 1, but not exactly
