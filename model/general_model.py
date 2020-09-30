from keras.models import Model
from keras.layers import Lambda, Input, BatchNormalization, Embedding
import keras.backend as K
import tensorflow as tf


def general_net(feat_model, identity_model, img_shape):
    
  IMAGE_H, IMAGE_W = img_shape 

  input_camA = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_camB = Input(shape=(IMAGE_H, IMAGE_W, 3))

  feat_A = feat_model(input_camA)
  feat_B = feat_model(input_camB)

  out_cls_A= identity_model(feat_A)
  out_cls_B= identity_model(feat_B)
    
  feat =  Lambda(lambda v: K.stack(v, axis=0), name='stacked_feats')([feat_A, feat_B])
    
  cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_A, out_cls_B])

  general_model = Model([input_camA, input_camB], [feat,cls], name='general_model') 
    
  return general_model

from keras.models import Sequential
def centernet(ident_num, embedding_shape):
  model = Sequential()
  model.add(Embedding(ident_num, embedding_shape))
  return model

def general_net_center(feat_model, identity_model, center_model, img_shape):
    
  IMAGE_H, IMAGE_W = img_shape 

  input_camA = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_camB = Input(shape=(IMAGE_H, IMAGE_W, 3))

  id_input_camA = Input((1,), name='target_input_camA')
  id_input_camB = Input((1,), name='target_input_camB')

  feat_A = feat_model(input_camA)
  feat_B = feat_model(input_camB)

  out_cls_A= identity_model(feat_A)
  out_cls_B= identity_model(feat_B)
    
  feat =  Lambda(lambda v: K.stack(v, axis=0), name='stacked_feats')([feat_A, feat_B])
    
  cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_A, out_cls_B])

  center_input = Lambda(lambda v: K.concatenate(v, axis=0), name = 'stacked_center_input')([id_input_camA, id_input_camB])

  # feats_for_center = Lambda(lambda v: K.concatenate(v, axis=0), name = 'stacked_feat_for_center')([feat_A, feat_B])
  feat_Aex = Lambda(lambda v: K.expand_dims(v, axis=1), name = 'expand_dimsA')(feat_A)
  feat_Bex = Lambda(lambda v: K.expand_dims(v, axis=1), name = 'expand_dimsB')(feat_B)

  centerA = center_model(id_input_camA)
  centerB = center_model(id_input_camB)


  l2_lossA = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_centerA')([feat_A, centerA])
  l2_lossB = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss_centerB')([feat_B, centerB])  

  general_model = Model([input_camA, input_camB, id_input_camA, id_input_camB], [feat, cls, l2_lossA, l2_lossB], name='general_model') 
    
  return general_model

    
