import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Lambda, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, LeakyReLU, Dropout
from keras.initializers import RandomNormal
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils
from keras.layers import BatchNormalization

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
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(ident_num, 
                    activation='softmax', 
                    use_bias = bias, 
                    bias_initializer = keras.initializers.Zeros(), 
                    kernel_initializer= keras.initializers.RandomNormal(stddev=0.001)))
    return model

def classification_layer_baseline(ident_num, BN = False):
    """Returns a keras Model with one FC layer

    Args:
            ident_num: size of FC layer
            bias: a Boolean specifying bias use

    Returns:
            model: Keras model 
    """  
    model = Sequential()
    if BN:
      print('Using kernel_initializer = he_uniform')
      model.add(BatchNormalization())
      model.add(Dense(
                  ident_num, 
                  activation='softmax',
                  use_bias = False,
                  kernel_initializer = keras.initializers.he_uniform()
                )
      )
    else:
      model.add(Dense(
                  ident_num, 
                  activation='softmax'
                )
      )
    return model

def classification_net(feat_model, identity_model, img_shape):
  """Returns a keras Model joining feat_model and dense layer

  Args:
          feat_model: keras Model for feature extracting
          identity_model: keras Model for classification
          img_shape: tuple specifying (h,w)

  Returns:
          model: Keras model 
  """  

  IMAGE_H, IMAGE_W = img_shape
  input_img = Input(shape=(IMAGE_H, IMAGE_W, 3))
  feat_ = feat_model(input_img) 
  out_cls = identity_model(feat_)
  cls_model = Model(input_img, out_cls, name='classification_net')

  return cls_model


def BNNeckForInference(feat_model, img_shape):
  """Returns a keras Model for batch norm features

  Args:
          feat_model: keras Model for feature extracting
          img_shape: tuple specifying (h,w)

  Returns:
          model: Keras model 
  """  

  IMAGE_H, IMAGE_W = img_shape
  input_img = Input(shape=(IMAGE_H, IMAGE_W, 3))
  feat_ = feat_model(input_img)
  bn = BatchNormalization()(feat_)
  bnnek_inf = Model(input_img, bn, name='BNNeckForInference')
  return bnnek_inf


# def general_net(feat_model, identity_model, img_shape):
#   """Returns a keras Model for training 
#   with classification and metric learning losses

#   Args:
#           feat_model: keras Model for feature extracting
#           identity_model: keras Model for classification
#           img_shape: tuple specifying (h,w)

#   Returns:
#           model: Keras model 
#   """  
#   IMAGE_H, IMAGE_W = img_shape 

#   input_camA = Input(shape=(IMAGE_H, IMAGE_W, 3))
#   input_camB = Input(shape=(IMAGE_H, IMAGE_W, 3))

#   id_input_camA = Input((1,), name='target_input_camA')
#   id_input_camB = Input((1,), name='target_input_camB')

#   feat_A = feat_model(input_camA)
#   feat_B = feat_model(input_camB)

#   out_cls_A= identity_model(feat_A)
#   out_cls_B= identity_model(feat_B)
    
#   feat =  Lambda(lambda v: K.stack(v, axis=0), name='stacked_feats')([feat_A, feat_B])
    
#   cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_A, out_cls_B])

#   general_model = Model([input_camA, input_camB], [feat,cls], name='general_model') 
    
#   return general_model