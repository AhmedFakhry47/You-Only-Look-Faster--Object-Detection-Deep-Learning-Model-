from __future__ import division
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensornets as nets
import numpy as np
from tensorflow.keras.utils import plot_model
from tensornets.references.yolo_utils import get_v2_boxes, v2_loss, v2_inputs
from tensornets.preprocess import darknet_preprocess as preprocess

def darkdepthsepconv(inputs, filters, kernel, name, lmbda=5e-4, dropout_rate=0):
  with tf.name_scope(name):
    x = tf.keras.layers.DepthwiseConv2D(kernel, depth_multiplier=1, padding='same', use_bias=False, name=name+'/sconv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=name+'/bnd')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False, name=name+'/conv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=True, scale=True, name=name+'/bns')(x)
    x = tf.nn.bias_add(x, tf.Variable(tf.random_normal([filters])), name= name+'bias_add')
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x

def meta(dataset_name='voc'):
  if dataset_name=='voc':
    bases = {}
    labels_voc={1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',9:'chair',10:'cow',11:'diningtable',12:'dog',13:'horse',14:'motorbike',15:'person',16:'pottedplant',17:'sheep',18:'sofa',19:'train',20:'tvmonitor'}
    bases['anchors'] =  [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                      8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

    bases.update({'num': 5})
    bases.update({'classes':20, 'labels': labels_voc})

  elif dataset_name == 'custom':
    bases = {}
    labels_custom = {0:'Stop',1:'TurnRight',2:'TurnLeft',3:'Person',4:'GreenLight',5:'RedLight',6:'None',7:'None',8:'None',9:'None',10:'None',10:'None',11:'None',12:'None',13:'None',14:'None',15:'None',16:'None',17:'None',18:'None',19:'None'}
    bases['anchors'] = [0.8675444162436549, 1.3149587563451777,1.2918364377182772, 1.9025029103608848,2.3540331196581197, 4.556490384615385,0.5772688356164384, 0.7527825342465754,3.6365404564315353, 5.26996887966805]

    bases.update({'num': 5})
    bases.update({'classes':20, 'labels': labels_custom})

  elif dataset_name == 'coco':
    bases = {}
    labels_coco = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
    labels_coco = {value:key for key, value in labels_coco.items()}
    bases['anchors'] = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                             5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    bases.update({'num': 5})
    bases.update({'classes':80, 'labels': labels_coco})
  return bases

def model(inputs, is_training=True, lmbda=5e-4, dropout_rate=0,type='coco'): 
  metas    =meta(type)
  N_classes=metas['classes']
  lmbda += 1e-10

  with tf.name_scope('stem'):
    x = stem = nets.MobileNet25(inputs, is_training=True, stem=True,  scope='stem') #bulding the model

  p = x.p

  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv7', lmbda=lmbda, dropout_rate=dropout_rate)
  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv8', lmbda=lmbda, dropout_rate=dropout_rate)

  p = darkdepthsepconv(p, 128, 1, name='genYOLOv2/conv5a', lmbda=lmbda, dropout_rate=dropout_rate)
  p = tf.reshape(p,[-1, 13,13,512], name='flat5a')
  x = tf.concat([p, x], axis=3, name='concat')

  x = darkdepthsepconv(x, 2048, 3, name='genYOLOv2/conv9', lmbda=lmbda, dropout_rate=dropout_rate)
  x = tf.keras.layers.Conv2D((N_classes+ 5) * 5, 1, kernel_regularizer=tf.keras.regularizers.l2(lmbda), padding='same', name='genYOLOv2/linear/conv_new')(x)
  x.aliases = []

  def get_boxes(*args, **kwargs):
  	return get_v2_boxes(metas, *args, **kwargs)
  x.get_boxes = get_boxes
  x.stem = stem
  x.inputs = [inputs]
  x.inputs += v2_inputs(x.shape[1:3], metas['num'], N_classes, x.dtype)
  if isinstance(is_training, tf.Tensor):
      x.inputs.append(is_training)
  x.loss = v2_loss(x, metas['anchors'], N_classes)
  def preprocess_(*args, **kwargs):
  	return preprocess(target_size=(416,416), *args, **kwargs)
  x.preprocess=preprocess_
  return x
