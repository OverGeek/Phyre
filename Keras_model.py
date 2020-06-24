import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from loss_functions import loss_fun_task2

num_epochs = 7
batch_size = 16

train_x_images = np.load('./data_x_images.npy')
train_x_vel = np.load('./data_x_vel.npy')
train_y = np.load('./one_hot-data_y.npy')

train_x_images, val_x_images, train_x_vel, val_x_vel, train_y, val_y = train_test_split(train_x_images,
                                                                                        train_x_vel,
                                                                                        train_y,
                                                                                        test_size=0.1)

model = load_model('Model_Task2', custom_objects={'loss_fun_task2': loss_fun_task2})

model.load_weights('./data_in_float/7-epochs.h5')

model.fit([train_x_images, train_x_vel], train_y, batch_size, num_epochs,
          shuffle=True,
          validation_data=([val_x_images, val_x_vel], val_y), verbose=True)

# model.save_weights('./7-epochs.h5')
