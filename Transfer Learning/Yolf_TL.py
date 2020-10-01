from __future__ import division
import matplotlib.pyplot as plt
import tensornets as nets
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import math
import coco
import cv2
import os

from Yolf_utils import *

#from tensorflow.python.training import py_checkpoint_reader
def get_variables(checkpoint_prefix):
  
  get_name = lambda x : x.name 
  stripper = lambda x : x.strip(':0')
  rem_tuble= lambda x : x[0]


  complete_variables   = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
  
  variables_names      = list(map(stripper,list(map(get_name,complete_variables))))
  checkpoint_variables = list(map(rem_tuble,tf.train.list_variables(checkpoint_prefix)))

  crossed_variables  = list(set(variables_names).intersection(set(checkpoint_variables)))
  indices = [variables_names.index(name) for name in crossed_variables] 
  return [complete_variables[i] for i in indices]



data_dir = '/lfs02/datasets/coco'
ann_dir  = '/home/alex054u4/data/nutshell/coco'


# Define the model hyper parameters

N_classes=80
is_training = tf.placeholder(tf.bool)

x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolo=model(x, lmbda=0, dropout_rate=0,is_training = is_training)


# Define an optimizer
epoch = tf.Variable(0,trainable=False,name="Epoch")
lr     = tf.Variable(1e-6,trainable=False,dtype=tf.float64)
lr_sch = tf.math.multiply(lr,tf.math.pow(tf.cast(0.5,tf.float64),tf.cast(tf.math.floormod(epoch,10),tf.float64)))
train  = tf.train.AdamOptimizer(lr, 0.9).minimize(yolo.loss)


#Check points for step training_trial_step
checkpoint_path   = "/home/alex054u4/data/nutshell/research/training_trial_YOLF1"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)


init_op     = tf.global_variables_initializer()
variables_can_be_restored = get_variables(checkpoint_prefix)#list(set(tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)).intersection(tf.train.list_variables(checkpoint_prefix))) 
train_saver = tf.train.Saver(variables_can_be_restored,max_to_keep=2)

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # for i in not_initialized_vars: # only for testing
    #    print(i.name)

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

    return

def evaluate_accuracy(data_type='te'):
  if (data_type  == 'tr') : acc_data   = coco.load(data_dir,ann_dir,'train2017')
  elif(data_type == 'te') : acc_data  = coco.load(data_dir,ann_dir, 'val2017')

  results = []
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolo, {x: yolo.preprocess(img),is_training: False})
    boxes    = yolo.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
  if (data_type  =='tr'):return coco.evaluate(results,data_dir,ann_dir,'train2017')
  elif (data_type=='te'):return coco.evaluate(results,data_dir,ann_dir,'val2017')

def lr_shoot(epoch,teacc,best_acc):
  '''
  A function to schedule the learning rate
  '''
  if (i == 10):sess.run(lr.assign(1e-3))
  if (i >= 15):
    if (teacc > best_acc):
      best_acc= teacc
      sess.run(epoch.assign(i))
      if(lr.eval() > 1e-8):
        sess.run(lr.assign(1e-8))
      else:
        sess.run(lr.assign(lr_sch))
    else:
      sess.run(lr.assign(1e-4))

  return best_acc,teacc 

with tf.Session() as sess:
  ckpt_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and 'ckpt' in f]
  if (len(ckpt_files)!=0):
    train_saver.restore(sess,checkpoint_prefix)
    initialize_uninitialized(sess)

  else:
    sess.run(init_op)
    sess.run(yolo.stem.pretrained())


  av_loss    = 0.0
  best_acc   = 0.0


  for i in tqdm(range(epoch.eval(),233)):    
    trains = coco.load_train(data_dir,ann_dir,'train2017', batch_size=64)
    losses     = 0.0
    for j,(imgs, metas) in enumerate(trains):
      if imgs is None: break

      metas.insert(0, yolo.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolo.loss],dict(zip(yolo.inputs, metas)))
      losses+=outs[-1]


    av_loss = losses/j  #AVG 


    if(math.isnan(av_loss)):
      print("NN output: \n", yolo)
      print('\n======================================================================================\n')
      print(tf.trainable_variables())

    print('\nepoch:',epoch.eval(),'lr: ',lr.eval(),'loss:',av_loss)

    #tracc_str,_     = evaluate_accuracy('tr')
    _,teacc = evaluate_accuracy('te')
    print ('\n')    

    best_acc,teacc = lr_shoot(i,teacc,best_acc)

    train_saver.save(sess,checkpoint_prefix)

    print ('dev set accuacy: ', teacc)
    print ('=================================================================================================================================================================================')
