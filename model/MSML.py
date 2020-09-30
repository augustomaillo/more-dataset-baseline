import tensorflow as tf

def msml(margin= 0.3):
    def squared_dist(A, B):


        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

        return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B

    def calculate_msml(y_true, y_pred):
        """Compute the margin sample mining loss.

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
            tf.where(tf.math.equal(mask,1), tf.ones_like(mask)*tf.reduce_max(dist), dist)
        )

        loss = tf.reduce_mean(
            tf.math.maximum(
                dist_max - dist_min + margin_tensor, 
                tf.constant([0], dtype=tf.float32)
            )
        )

        return loss
    return calculate_msml