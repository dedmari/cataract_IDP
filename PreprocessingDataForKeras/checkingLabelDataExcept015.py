import numpy as np
dataset = np.loadtxt("../cataractsProject/train-labels/train01WithoutHeading.csv", delimiter=",")
for i in range(dataset.shape[0]):
	for j in range(dataset.shape[1]):
		if (dataset[i][j] != 0) and (dataset[i][j] != 1) and (dataset[i][j] != 0.5):
			print dataset[i][j]
			print i,j

print "finished"
