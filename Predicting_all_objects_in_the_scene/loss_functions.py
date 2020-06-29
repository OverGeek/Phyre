import keras
import tensorflow as tf

num_classes = 100


def loss_fun_single_object(y_true, y_pred):
    vx_true = y_true[:, :num_classes]
    vy_true = y_true[:, num_classes: 2 * num_classes]
    theta_true = y_true[:, 2 * num_classes:]

    vx_pred = y_pred[:, :num_classes]
    vy_pred = y_pred[:, num_classes: 2 * num_classes]
    theta_pred = y_true[:, 2 * num_classes:]

    loss_x = keras.losses.categorical_crossentropy(vx_true, vx_pred)
    loss_y = keras.losses.categorical_crossentropy(vy_true, vy_pred)
    loss_theta = keras.losses.categorical_crossentropy(theta_true, theta_pred)

    loss = tf.reduce_mean(tf.add(tf.add(loss_x, loss_y), loss_theta), axis=0)

    return loss


def loss_func(y_true, y_pred, num_objects=4):
    loss = tf.zeros((1,))
    for i in range(num_objects):
        loss = tf.add(loss, loss_fun_single_object(y_true[num_objects * i: num_objects * (i + 1)],
                                                   y_pred[num_objects * i: num_objects * (i + 1)]))

    return tf.divide(loss, num_objects)
