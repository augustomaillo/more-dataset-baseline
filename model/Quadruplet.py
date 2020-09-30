from keras.models import Model
from keras.layers import Lambda, Input, BatchNormalization
import keras.backend as K
import tensorflow as tf


# def euclidean_distance(v):
#     x, y = v
#     return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# def quadruplet_net(feat_model, identity_model, img_shape):
  
#   IMAGE_H, IMAGE_W = img_shape 

#   input_anchor_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
#   input_anchor_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
#   input_pos = Input(shape=(IMAGE_H, IMAGE_W, 3))
#   input_neg = Input(shape=(IMAGE_H, IMAGE_W, 3))
  
#   # computing the embedded feature space
#   feat_anchor_pos = feat_model(input_anchor_pos) 
#   feat_anchor_neg = feat_model(input_anchor_neg)
#   feat_pos = feat_model(input_pos)
#   feat_neg = feat_model(input_neg)

#   # classification  
#   out_cls_pos1 = identity_model(feat_anchor_pos)
#   out_cls_neg1 = identity_model(feat_anchor_neg)
#   out_cls_pos2 = identity_model(feat_pos)
#   out_cls_neg2 = identity_model(feat_neg)

#   positive_dist = Lambda(euclidean_distance, name='pos_dist')([feat_anchor_pos, feat_pos])
#   negative_dist = Lambda(euclidean_distance, name='neg_dist')([feat_anchor_neg, feat_neg])

#   dist = Lambda(lambda v: K.stack(v, axis=1), name='stacked_dists')([positive_dist, negative_dist])
#   cls = Lambda(lambda v: K.stack(v, axis=1), name='stacked_cls')([out_cls_pos1, out_cls_neg1, out_cls_pos2, out_cls_neg2])
#   quad_model = Model([input_anchor_pos, input_anchor_neg, input_pos, input_neg], [dist,cls], name='quadruplet_siamese')
#   return quad_model


def quadruplet_accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0]) # distância entre o par de positivas é menor que o entre o par de amostras negativas.


def quadruplet_loss(margin= 0.3):

    def squared_dist(A, B):
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

    def calculate_quadloss(y_true, y_pred):
        """Compute the quadruplet loss.

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

        dist_negpair = tf.math.reduce_min(
            tf.transpose(
              tf.gather(
                  tf.where(tf.math.equal(mask,1), tf.ones_like(mask)*tf.reduce_max(dist), dist), 
                  tf.where(tf.math.equal(dist, tf.expand_dims(dist_min, axis=-1)))[:,1],
                  axis=1
                )
            ), axis=0
        )

        loss = tf.reduce_mean(
            tf.math.maximum(
                dist_max - dist_min + margin_tensor, 
                tf.constant([0], dtype=tf.float32)
            )
        ) + tf.reduce_mean(
            tf.math.maximum(
                dist_max - dist_negpair + margin_tensor, 
                tf.constant([0], dtype=tf.float32)
            )
        )
        
        return loss
    return calculate_quadloss