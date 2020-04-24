from keras.models import Model
from keras.layers import Lambda, Input, BatchNormalization
import keras.backend as K

def euclidean_distance(v):
    x, y = v
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def quadruplet_net(feat_model, identity_model, img_shape):
  
  IMAGE_H, IMAGE_W = img_shape 

  input_anchor_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_anchor_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
  
  # computing the embedded feature space
  feat_anchor_pos = feat_model(input_anchor_pos) 
  feat_anchor_neg = feat_model(input_anchor_neg)
  feat_pos = feat_model(input_pos)
  feat_neg = feat_model(input_neg)

  # classification  
  out_cls_pos1 = identity_model(feat_anchor_pos)
  out_cls_neg1 = identity_model(feat_anchor_neg)
  out_cls_pos2 = identity_model(feat_pos)
  out_cls_neg2 = identity_model(feat_neg)

  # normalizing the feature vector
  # feat_anchor_pos = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_anchor_pos)
  # feat_anchor_neg = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_anchor_neg)


  # feat_pos = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_pos)
  # feat_neg = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_neg) 


  positive_dist = Lambda(euclidean_distance, name='pos_dist')([feat_anchor_pos, feat_pos])
  negative_dist = Lambda(euclidean_distance, name='neg_dist')([feat_anchor_neg, feat_neg])

  dist = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists')([positive_dist, negative_dist])
  cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_pos1, out_cls_neg1, out_cls_pos2, out_cls_neg2])
  quad_model = Model([input_anchor_pos, input_anchor_neg, input_pos, input_neg], [dist,cls], name='quadruplet_siamese')
  return quad_model


def quadruplet_accuracy(y_true, y_pred):
    
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0]) # distância entre o par de positivas é menor que o entre o par de amostras negativas.

# Definindo a função de loss
def quadruplet_loss(margin= 0.3):
  def _quadruplet_loss(y_true, y_pred):
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

  return _quadruplet_loss