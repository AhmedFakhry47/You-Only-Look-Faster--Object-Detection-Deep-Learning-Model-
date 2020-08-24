'''
Runtime script:
1-Takes checkpoint path
2-
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensornets as nets
#from YOLF_utils import *
import numpy as np
import sys
import cv2
import os

def darknet_preprocess(x, target_size=(416,416)):
  if target_size is None or target_size[0] is None or target_size[1] is None:
      y = x.copy()
  else:
      h, w = target_size
      assert cv2 is not None, 'resizing requires `cv2`.'
      y = np.zeros((h, w, 3))
      y = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
  y = y[np.newaxis, :, :, ::-1].astype(np.float32)
  y /= 255.
  return y

C1=[238, 72, 58, 24, 203, 230, 54, 167, 246, 136, 106, 95, 226, 171, 43, 159, 231, 101, 65, 157]
C2=[122, 71, 173, 32, 147, 241, 53, 197, 228, 164, 4, 209, 175, 223, 176, 182, 48, 3, 70, 13]
C3=[148, 69, 133, 41, 157, 137, 125, 245, 89, 85, 162, 43, 16, 178, 197, 150, 13, 140, 177, 224]
idx_to_labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

def visualize_img(img,bboxes,thickness,name):
  img=img.reshape(img.shape[1],img.shape[1],3)
  for c, boxes_c in enumerate(bboxes):
    for b in boxes_c:
      ul_x,ul_y = b[0],b[1]
      br_x,br_y = b[2],b[3]
      ul_x, ul_y=(min(max(int(ul_x),0),415),min(max(int(ul_y),0),415))
      br_x, br_y=(min(max(int(br_x),0),415),min(max(int(br_y),0),415))

      color_class=(C1[c], C2[c], C3[c])
      img=cv2.rectangle(img, (ul_x, ul_y), (br_x, br_y), color=color_class, thickness=3) 
      label = '%s: %.2f' % (idx_to_labels[c], b[-1]) 
      labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
      ul_y = max(ul_y, labelSize[1]) 
      img=cv2.rectangle(img, (ul_x, ul_y - labelSize[1]), (ul_x + labelSize[0], ul_y + baseLine),color_class, cv2.FILLED) 
      img=cv2.putText(img, label, (ul_x, ul_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)) 

  cv2.imwrite(name+'.jpg', img)
  return img

def main(checkpoint_path,video_path):
	training_video 		= cv2.VideoCapture(video_path)
	checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")


	# Put the model here
	x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
	#yolf=model(x, lmbda=0, dropout_rate=0)
	yolf = nets.YOLOv2VOC(x,is_training=False,classes=20)

	#saver to load checkpoints
	#train_saver 			= tf.train.Saver(max_to_keep=2)

	#Output video
	##Define the codec and create the output video object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (416,416))

	with tf.Session() as sess:
		#train_saver.restore(sess,checkpoint_prefix)
		sess.run(tf.global_variables_initializer())
		sess.run(yolf.pretrained())

		while(training_video.isOpened()):
		    ret, frame = training_video.read()

		    #If successfuly the video frame is retreived
		    if(ret == True):
			    frame = darknet_preprocess(frame, target_size=(416,416))

			    predictions = sess.run(yolf, {x: frame})
			    boxes   		= yolf.get_boxes(predictions, frame.shape[1:3])
			    #Write the current frame in the video
			    out.write(visualize_img((frame*255).astype(np.uint8),boxes,5,'img'))

		#Close video files
		training_video.release()
		out.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	checkpoint_path = sys.argv[1]
	video_path 			= sys.argv[2]
	main(checkpoint_path,video_path)
