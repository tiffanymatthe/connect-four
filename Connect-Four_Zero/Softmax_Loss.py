import tensorflow as tf


def softmax_cross_entropy_with_logits(y_true, y_pred):

    p = y_pred # probability distribution, -1 to 1?
    pi = y_true # probability distribution with values from 0 to 1

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred"")

    pi = tf.cast(pi, dtype=tf.float32)
    p = tf.cast(p, dtype=tf.float32)

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)
    negatives = tf.fill(tf.shape(pi), -100.0) 
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

    return loss

# if __name__ == "__main__":
#     y_true = ([0,0,3,0,180,0,0]
#     y_pred = [0.1, 0.9, 0,40,0,0,0]
#     print(softmax_cross_entropy_with_logits(y_true, y_pred))