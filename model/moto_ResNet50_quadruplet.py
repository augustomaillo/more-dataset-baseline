import scipy.spatial.distance
import keras
from keras.models import Model
from keras.layers import Lambda, Input
import keras.backend as K
from model import moto_ResNet50

# Definindo modelo do triplet loss 
def euclidean_distance(v):
    x, y = v
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def quadruplet_net(img_shape , feat_model, feat_graph, identity_model, ident_graph):
  
  IMAGE_H, IMAGE_W = img_shape 

  input_anchor_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_anchor_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
  
  with feat_graph.as_default():
    feat_anchor_pos = feat_model(input_anchor_pos)
  
  with feat_graph.as_default():    
    feat_anchor_neg = feat_model(input_anchor_neg)
  
  with feat_graph.as_default():  
    feat_pos = feat_model(input_pos)
    
  with feat_graph.as_default():  
    feat_neg = feat_model(input_neg)
    
  with ident_graph.as_default():  
    out_cls_pos1 = identity_model(feat_anchor_pos)
  
  with ident_graph.as_default():  
    out_cls_neg1 = identity_model(feat_anchor_neg)
  
  with ident_graph.as_default():
    out_cls_pos2 = identity_model(feat_pos)
  
  with ident_graph.as_default():  
    out_cls_neg2 = identity_model(feat_neg)

  feat_anchor_pos = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_anchor_pos)
  feat_anchor_neg = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_anchor_neg)
  feat_pos = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_pos)
  feat_neg = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_neg)

  positive_dist = Lambda(euclidean_distance, name='pos_dist')([feat_anchor_pos, feat_pos])
  negative_dist = Lambda(euclidean_distance, name='neg_dist')([feat_anchor_neg, feat_neg])

  dist = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists')([positive_dist, negative_dist])
  cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_pos1, out_cls_neg1, out_cls_pos2, out_cls_neg2])

  quad_model = Model([input_anchor_pos, input_anchor_neg, input_pos, input_neg], [dist,cls], name='qadruplet_siamese')
  return quad_model


