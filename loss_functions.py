import keras
import tensorflow as tf

num_classes = 100


def loss_fun_task2(y_true, y_pred):
    vx_true = y_true[:, :num_classes]
    vy_true = y_true[:, num_classes:]
    vx_pred = y_pred[:, :num_classes]
    vy_pred = y_pred[:, num_classes:]

    loss_x = keras.losses.categorical_crossentropy(vx_true, vx_pred)
    loss_y = keras.losses.categorical_crossentropy(vy_true, vy_pred)

    loss = tf.reduce_mean(tf.add(loss_x, loss_y), axis=0)

    return loss
