from keras.models import Model
from keras.layers import Lambda, Input, BatchNormalization
import keras.backend as K
import tensorflow as tf


def triplet_net(feat_model, identity_model, img_shape):
    
  IMAGE_H, IMAGE_W = img_shape 

  input_camA = Input(shape=(IMAGE_H, IMAGE_W, 3))
  input_camB = Input(shape=(IMAGE_H, IMAGE_W, 3))

  feat_A = feat_model(input_camA)
  feat_B = feat_model(input_camB)

  out_cls_A= identity_model(feat_A)
  out_cls_B= identity_model(feat_B)

#   feat_A = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_A)
#   feat_B = Lambda(lambda  x: K.l2_normalize(x,axis=1))(feat_B)
    
#   ex_clsA = Lambda(lambda v: tf.expand_dims(v, axis=0), name='expandA')(out_cls_A)
#   ex_clsB = Lambda(lambda v: tf.expand_dims(v, axis=0), name='expandB')(out_cls_B)
    
  feat =  Lambda(lambda v: K.stack(v, axis=0), name='stacked_feats')([feat_A, feat_B])
    
  cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_A, out_cls_B])

  tri_model = Model([input_camA, input_camB], [feat,cls], name='triplet_siamese') 
    
  return tri_model

def triplet_loss(margin= 0.3):
    def squared_dist(A, B):
#   assert A.shape.as_list() == B.shape.as_list()

        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

    def calculate_triloss(y_true, y_pred):
        """Compute the triplet loss.

        Args:
            y_true: tensor of shape (batch_size, 2, ident_num)
            y_pred: tensor of shape (2, batch_size, embedding_dims) 

        Returns:
            loss
        """  
        y_preda = y_pred[0]
        y_predb = y_pred[1]
        y_truea = y_true[:,0,:]
        y_trueb = y_true[:,1,:]
        
        margin_tensor = tf.constant([margin])

        
        dist = squared_dist(y_preda, y_predb)
        dist = tf.sqrt(dist + 1e-6) 
        
        la = tf.where(tf.math.equal(y_truea, 1))
        lb = tf.where(tf.math.equal(y_trueb, 1))    
        mask = tf.zeros((tf.shape(la)[0], tf.shape(lb)[0]))

        tempA = tf.tile(tf.expand_dims(tf.transpose(la[:,1]), axis=1), tf.stack([tf.constant(1), tf.shape(lb)[0]], axis=0)  )
        tempB = tf.tile(tf.expand_dims((lb[:,1]), axis=0), tf.stack([tf.shape(la)[0], tf.constant(1)], axis=0)  )

        mask = tf.where(tf.math.equal(tempA, tempB), tf.ones_like(mask), mask)

        dist_max = tf.math.reduce_max(
            tf.math.multiply(dist, tf.cast(tf.math.equal(mask,1), dtype=tf.float32)), 
            axis=1)

        dist_min = tf.math.reduce_min(
            tf.where(tf.math.equal(mask,1), tf.ones_like(mask)*tf.reduce_max(dist), dist), 
            axis=1)

        loss = tf.reduce_mean(
            tf.math.maximum(
                dist_max - dist_min + margin_tensor, 
                tf.constant([0], dtype=tf.float32)
            )
        )

        return loss
    return calculate_triloss
 


    
