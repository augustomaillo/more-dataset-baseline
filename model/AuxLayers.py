import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Lambda, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, LeakyReLU, Dropout
from keras.initializers import RandomNormal
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils

def embedding(featnet, bias = True):
    """Returns a keras Model joining featnet with 2 FC layers

    Args:
            featnet: a keras Model to be joined
            bias: a Boolean specifying bias use

    Returns:
            model: Keras model 
    """  
    x = featnet.output
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(1024,
              activation='relu',
              use_bias = bias, 
              bias_initializer = keras.initializers.Zeros(), 
              kernel_initializer= keras.initializers.RandomNormal(stddev=0.001))(x)
    
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(128, activation='relu',
              use_bias = bias, 
              bias_initializer = keras.initializers.Zeros(),
              kernel_initializer= keras.initializers.RandomNormal(stddev=0.001))(x)
    model = Model(featnet.input, x)
    return model

def classification_layer(ident_num, bias=True):
    """Returns a keras Model with one FC layer

    Args:
            ident_num: size of FC layer
            bias: a Boolean specifying bias use

    Returns:
            model: Keras model 
    """  
    model = Sequential()
    model.add(BatchNormalization()) # Prates (03/12)
    model.add(Dropout(0.5))
    model.add(Dense(ident_num, 
                    activation='softmax', 
                    use_bias = bias, 
                    bias_initializer = keras.initializers.Zeros(), 
                    kernel_initializer= keras.initializers.RandomNormal(stddev=0.001)))
    return model

def classification_net(feat_model, identity_model, img_shape):
  IMAGE_H, IMAGE_W = img_shape
  input_img = Input(shape=(IMAGE_H, IMAGE_W, 3))
  feat_ = feat_model(input_img) 
  out_cls = identity_model(feat_)
  cls_model = Model(input_img, out_cls, name='classification_net')

  return cls_model