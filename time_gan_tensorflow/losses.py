import tensorflow as tf

@tf.function
def mean_squared_error(y_true, y_pred):
    """
    Mean squared error, used for calculating the supervised loss and the reconstruction loss.
    """
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(y_true, y_pred)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

@tf.function
def binary_crossentropy(y_true, y_pred):
    """
    Binary cross-entropy, used for calculating the unsupervised loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
