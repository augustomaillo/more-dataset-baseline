import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Lambda, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, LeakyReLU, Dropout
from keras.initializers import RandomNormal
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils

def embedding(featnet, bias = True):
    x = featnet.output
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(1024, activation='relu')(x) #, use_bias = bias, bias_initializer = keras.initializers.Zeros(), kernel_initializer= keras.initializers.he_normal(seed=None))(x)
    x = BatchNormalization()(x) 
    x = Dropout(0.5)(x) 
    x = Dense(128, activation='relu')(x) #, use_bias = bias, bias_initializer = keras.initializers.Zeros(), kernel_initializer= keras.initializers.he_normal(seed=None))(x)
    model = Model(featnet.input, x)
    return model

def classification_layer(ident_num, bias=True):
    model = Sequential()
    model.add(BatchNormalization()) # Prates (03/12)
    model.add(Dropout(0.5))
    model.add(Dense(ident_num, activation='softmax') )#, use_bias = bias, bias_initializer = keras.initializers.Zeros(), kernel_initializer= keras.initializers.he_normal(seed=None)))
    return model