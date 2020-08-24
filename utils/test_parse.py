import yolf_data
import numpy as np
import cv2

data_dir = '/run/media/enihcam/New/Downloads/New-Code/TRAINdevKit/train'

trains = yolf_data.load_train(data_dir,'trainval', batch_size=48)


for j,(imgs, metas) in enumerate(trains):
	meta = metas[-1]
	img = imgs[0]	
	break

print(metas[-1])
print(imgs[0])