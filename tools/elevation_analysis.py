import numpy as np
import matplotlib.pyplot as plt
import os

path = "./analysis/SingleFiles/Dataset_Generation/"
data = os.listdir("./analysis/SingleFiles/Dataset_Generation")
true_positives = []
false_positives = []
total_positives = []
elevation_angle = []
for cnt,single_file in enumerate(data):
	print("%d - %s"%(cnt,single_file))
while True:
	x = input("Which analysis do you want to include?  ")
	if x == 'y':
		break
	with open(path+data[int(x)]+'/analysis.log') as f:
		f = f.readlines()
		for line in f:
			if 'True positive' in line:
				true_positives.append(int(line.split(" ")[-1]))
			if 'False positive' in line:
				false_positives.append(int(line.split(" ")[-1]))
	total_positives.append(true_positives[-1]+false_positives[-1])
	
	elev = data[int(x)].split("_")
	elevation_angle.append(int(elev[-1]))

plt.plot(elevation_angle,true_positives)
plt.ylim([0,max(total_positives)+2])
plt.xlabel("Elevation angle")
plt.ylabel("True positives")

plt.show()
