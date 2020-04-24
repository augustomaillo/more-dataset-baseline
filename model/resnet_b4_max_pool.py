# Definindo o modelo deep learning
import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Lambda, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, LeakyReLU, Dropout
from keras.initializers import RandomNormal
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils


def feat_net(img_shape):

  IMAGE_H, IMAGE_W = img_shape
  input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
  # Rede Base
  base_model = keras.applications.resnet50.ResNet50(include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=(IMAGE_H,IMAGE_W,3))

  # base_model.summary()
    
  featMap = base_model.output
  # feat = GlobalAveragePooling2D()(featMap)

  # b4 max pooling
  feat = GlobalAveragePooling2D()(base_model.get_layer(index = 142).output)

  model = Model(input=base_model.input, output = feat)

  return model