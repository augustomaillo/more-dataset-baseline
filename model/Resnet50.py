# Definindo o modelo deep learning
import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Lambda, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, LeakyReLU, Dropout
from keras.initializers import RandomNormal
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils

def classification_net(ident_num, bias = True):
  model = Sequential()
  
  model.add(BatchNormalization()) 
  model.add(Dropout(0.5)) 

  model.add(Dense(256, activation='relu'))
  model.add(BatchNormalization()) 
  model.add(Dropout(0.5)) 

  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization()) 
  model.add(Dropout(0.5)) 
  
  model.add(Dense(ident_num, activation='softmax', use_bias = bias))



  return model

def feat_net(img_shape):
  """Returns a keras Model for feature extract

  Args:
          img_shape: tuple (h,w)

  Returns:
          model: Keras model 
  """  

  IMAGE_H, IMAGE_W = img_shape
  input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))

  base_model = keras.applications.resnet50.ResNet50(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(IMAGE_H,IMAGE_W,3))

  featMap = base_model.output
  feat = GlobalAveragePooling2D()(featMap)

  model = Model(input=base_model.input, output = feat)

  return model