'''
script to calculate anchor boxes 
1-We will find the distribution of bounding boxes coordinates relative to image size (416,416)
2-will find random 5 points that covers this distribution
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random
import os


data_dir = '/run/media/enihcam/New/Downloads/New-Code/data'
file_names = ['Training0-OUTPUT.txt','Training1-OUTPUT.txt','Training2-OUTPUT.txt']

relative_widths = []
relative_hights = []

AB  = [] #Anchorboxes

for file in file_names:
	current = open(os.path.join(data_dir,file),'r')

	for line in current.readlines():
		line  = line.strip()
		parts = line.split('/')

		num_boxes = len(parts)-1

		for i in range(0,num_boxes):
			coordinates = parts[i+1].split(',')
			width  = int(coordinates[2]) - int(coordinates[0])	#ulx - drx
			hight  = int(coordinates[3]) - int(coordinates[1])	#uly - dry

			relative_w = width/32	#32 is the feature size
			relative_h = hight/32

			relative_widths.append(relative_w)
			relative_hights.append(relative_h)

Fig,ax = plt.subplots()
ax.set_xlim(0,13)
ax.set_ylim(0,13)

plt.scatter(relative_widths,relative_hights)


'''
We are gonna calculate the anchor boxes as an inverse problem of the K-means clustering algorithm
Meaning that instead of calculating clusters and cluster changes we will focus on calculating 
the centroids. Because that what we care about.
'''

points = [[i,j] for i,j in zip(relative_widths,relative_hights)]
random.shuffle(points)

centroids = random.sample(points,5)

num_anchorboxes = 5

#Assume number of clusters = 5
#We are gonna create 5 lists to contain the points that belong to each cluster

clusters = {i:[] for i in range(0,num_anchorboxes)}

#K means clustering algorithm 
num_changes = 0

#Calculate distance function
distance = lambda p0,p1: np.sqrt((p1[0]-p0[0])**2+(p1[1]-p0[1])**2)

while(True):
	#First assign clusters 
	for point in points:
		cl_id = np.argmin([distance(point,j) for j in centroids])

		if point not in clusters[cl_id]:
			num_changes += 1
			clusters[cl_id].append(point)

	if num_changes == 0:
		break
	num_changes = 0

	#Recalculate centroids 
	for i in range(0,len(centroids)):
		points_mean	 = np.mean(np.array(clusters[i]),axis=0)
		centroids[i] = [points_mean[0],points_mean[1]]


print('Anchorboxes : \n',centroids)
centroids = np.array(centroids).T
plt.scatter(centroids[0],centroids[1])
plt.xlabel('Relative width')
plt.ylabel('Relative hight')
plt.show()

