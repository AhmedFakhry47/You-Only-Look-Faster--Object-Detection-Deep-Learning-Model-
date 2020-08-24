import xml.etree.ElementTree as etree 
import numpy as np
import json
import sys
import cv2
import os

Id_to_class = {'0':'Stop','1':'TurnRight','2':'TurnLeft','3':'Person','4':'GreenLight','5':'RedLight'}

def map_imgcounter(number):
	number = str(number)
	return '0'*(5-len(number))+number

def map_fileline(line):
	parts 	  = line.split('/')
	num_boxes = len(parts)-1

	boxes = {} 

	for i in range(0,num_boxes):
		boxes[i]= []
		coordinates = parts[i+1].split(',')
		boxes[i].append(coordinates[0])
		boxes[i].append(coordinates[1])
		boxes[i].append(coordinates[2])
		boxes[i].append(coordinates[3])
		boxes[i].append(coordinates[4])

	return num_boxes,boxes

def store_img(img,data_dir,file_name):
	img     = img.reshape(img.shape[1],img.shape[1],3)
	img_dir = os.path.join(data_dir,file_name)
	cv2.imwrite(img_dir+'.jpg', img)
	return 

def open_file(file_path):
	file = open(file_path,'r')
	while(True):
		for line in file.readlines():
			yield line.strip()

def Imagesets_file(file_path,filename,imgs_c):
	Imagset_file = open(os.path.join(file_path,filename+'.txt'),"a")

	for num in range(0,imgs_c):
		Imagset_file.write(map_imgcounter(num))
		Imagset_file.write('\n')
	Imagset_file.close()

def create_XML(meta,data_dir,xmlfile):
	root = etree.Element("annotation") 
	num_boxes,boxes = map_fileline(meta)
	if (num_boxes ==0):
		return -1
	for box_num in range(0,num_boxes):
		objects = etree.Element("object")
		root.append(objects) 

		name = etree.SubElement(objects, "name") 
		name.text = Id_to_class[boxes[box_num][-1]]

		pose = etree.SubElement(objects, "pose") 
		pose.text = "Unspecified"

		truncated = etree.SubElement(objects, "truncated") 
		truncated.text = "0"

		difficult = etree.SubElement(objects, "difficult") 
		difficult.text = "0"

		bndbox    = etree.SubElement(objects,"bndbox")
		xmin	  = etree.SubElement(bndbox,"xmin") 
		xmin.text = boxes[box_num][0]

		ymin	  = etree.SubElement(bndbox,"ymin") 
		ymin.text = boxes[box_num][1] 

		xmax	  = etree.SubElement(bndbox,"xmax") 
		xmax.text = boxes[box_num][2]

		ymax	  = etree.SubElement(bndbox,"ymax") 
		ymax.text = boxes[box_num][3]

	tree = etree.ElementTree(root)   
	filedir = os.path.join(data_dir,xmlfile+'.xml')
	with open (filedir, "wb") as file : 
		tree.write(file) 
	return 0

def main(videos_path):
	imgs_dir = os.path.join(os.getcwd(),'JPEGImages')
	anns_dir = os.path.join(os.getcwd(),'Annotations')

	sets_dir = os.path.join(os.getcwd(),'ImageSets')
	os.mkdir(os.path.join(sets_dir,'Main'))
	sets_dir = os.path.join(sets_dir,'Main')

	imgs_c	 = 0

	#Create Root for XMl Annotations file
	data = {'videos':[],'frames':[]}

	for dir_name,_,files in os.walk(videos_path):
		for file in files:
			if(str(file).split('.')[-1] == "mp4"):
				data['videos'].append(os.path.join(dir_name,file))
				frames_file = str(file).split('.')[0]+'-OUTPUT.txt'
				data['frames'].append(os.path.join(dir_name,frames_file))


	for video_path,frames_path in zip(data['videos'],data['frames']):	
		training_video 		= cv2.VideoCapture(video_path)
		frames_data  		= open_file(frames_path)

		while(training_video.isOpened()):
			ret, frame = training_video.read()
			if(ret == True):
				#Save the Image and It's annotations 
				img_id = map_imgcounter(imgs_c)
				status = create_XML(next(frames_data),anns_dir,img_id)
				if (status == -1):
					continue
				store_img(frame*255,imgs_dir,img_id)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				imgs_c += 1
			else:
				break
		#Close video files
		training_video.release()
		cv2.destroyAllWindows()

	#Create Imageset file
	Imagesets_file(sets_dir,'trainval',imgs_c)



if __name__ == '__main__':

	videos_path = sys.argv[1]

	if ('TRAINdevKit' not in os.listdir() ):
		os.mkdir('TRAINdevKit')
		os.chdir(os.path.join(os.getcwd(),'TRAINdevKit'))
		os.mkdir('train')
		os.chdir(os.path.join(os.getcwd(),'train'))
		os.mkdir('Annotations')
		os.mkdir('ImageSets')
		os.mkdir('JPEGImages')
	main(videos_path)

