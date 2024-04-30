from matplotlib import pyplot as plt
#test
x1 = [0, 5, 10]
y1 = [5, 15, 70]

x2 = [0, 5, 10]
y2 = [8, 15, 97]

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
ax.plot(x1, y1, label = 'Monte Carlo')
ax.plot(x2, y2, label = 'IS Count')
plt.show()