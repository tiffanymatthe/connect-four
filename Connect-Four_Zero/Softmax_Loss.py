import tensorflow as tf


def softmax_cross_entropy_with_logits(y_true, y_pred):

    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.int64)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0) 
    p = tf.where(where, negatives, p)
    pi = tf.constant(pi, dtype=tf.float32)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

    return loss

# if __name__ == "__main__":
#     y_true = [0,0,3,0,180,0,0]
#     y_pred = [0.1, 0.9, 0,40,0,0,0]
#     print(softmax_cross_entropy_with_logits(y_true, y_pred))