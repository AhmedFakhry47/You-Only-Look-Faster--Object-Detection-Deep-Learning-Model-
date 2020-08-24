from __future__ import division
import yolfnets as nets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
import random
import cv2
from copy import copy, deepcopy
from pathlib import Path
import os
import time 
from datetime import timedelta
from tqdm import tqdm
#import zipfile
import tarfile
import shutil
import wget
import sys
import voc
from Yolf_utils import *


voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'

# Define the model hyper parameters
N_classes=20
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolo=model(x, lmbda=0, dropout_rate=0)
# Define an optimizer
epoch = tf.Variable(0,trainable=False,name="Epoch")

lr     = tf.Variable(1e-3,trainable=False,dtype=tf.float64)
lr_sch = tf.math.multiply(lr,tf.math.pow(tf.cast(0.5,tf.float64),tf.math.divide(epoch,10)))
train  = tf.train.AdamOptimizer(lr, 0.9).minimize(yolo.loss)


#Check points for step training_trial_step
checkpoint_path   = "/home/alex054u4/data/nutshell/training_trial_YOLF_GOLD"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)


init_op     = tf.global_variables_initializer()
train_saver = tf.train.Saver(max_to_keep=2)

def evaluate_accuracy(data_type='tr'):
  if (data_type  == 'tr'): acc_data  = voc.load(voc_dir % 2007,'trainval',total_num =48)
  elif(data_type == 'te') : acc_data  = voc.load(voc_dir % 2007, 'test', total_num=48)

  #print('Train Accuracy: ',voc.evaluate(boxes, voc_dir % 2007, 'trainval'))
  results = []
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolo, {x: yolo.preprocess(img),is_training: False})
    boxes=yolo.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
  if (data_type  =='tr'):return voc.evaluate(results, voc_dir % 2007, 'trainval')
  elif (data_type=='te'):return voc.evaluate(results, voc_dir % 2007, 'test')


with tf.Session() as sess:
  ckpt_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and 'ckpt' in f]
  if (len(ckpt_files)!=0):
    train_saver.restore(sess,checkpoint_prefix)
  else:
    sess.run(init_op)
    sess.run(yolo.stem.pretrained())


  losses     = 0.0
  av_loss    = 0.0
  best_acc   = 0.0
  best_epoch = 0
  for i in tqdm(range(epoch.eval(),233)):    
    trains = voc.load_train([voc_dir % 2007, voc_dir % 2012],'trainval', batch_size=48)

    for j,(imgs, metas) in enumerate(trains):
      # `trains` returns None when it covers the full batch once
      if imgs is None: break
      metas.insert(0, yolo.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolo.loss],dict(zip(yolo.inputs, metas)))
      losses+=outs[-1]


    av_loss = 0.9*av_loss + 0.1*(losses/j)  #Moving average for loss 


    if(math.isnan(av_loss)):
      print("NN output: \n", yolo)
      print('\n======================================================================================\n')
      print(tf.trainable_variables())

    print('\nepoch:',step.eval(),'lr: ',lr.eval(),'loss:',av_loss)

    tracc_str,_     = evaluate_accuracy('tr')
    teacc_str,teacc = evaluate_accuracy('te')
    print ('\n')    

    if(i%10 == 0):
      if (teacc > best_acc):
        best_acc= acc
        sess.run(epoch.assign(i))
        sess.run(lr.assign(lr_sch))
        train_saver.save(sess,checkpoint_prefix)
      else:
        sess.run(lr.assign(1e-4))

    print ('highest training accuacy:', acc_best, 'at epoch:', best_epoch, '\n')
    print ('=================================================================================================================================================================================')