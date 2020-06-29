import numpy as np
from sklearn.model_selection import train_test_split
from model import get_model
from keras.callbacks import ModelCheckpoint
import pickle

train_x_images = np.load('./Transformed_dataset/data_x_images.npy')
train_x_features = np.load('./Transformed_dataset/data_x_features.npy')
train_y = np.load('./Transformed_dataset/one_hot-data_y.npy')

batch_size = 16
num_epochs = 30

num_objects = train_x_features.shape[1]

model = get_model()

train_x_images, val_x_images, train_x_features, val_x_features, train_y, val_y = train_test_split(train_x_images,
                                                                                                  train_x_features,
                                                                                                  train_y,
                                                                                                  test_size=0.1)

# model.load_weights('./data_in_float/7-epochs.h5')

train_input = []
for i in range(num_objects):
    train_input.append(train_x_images)
    train_input.append(train_x_features[:, i, :])

val_input = []
for i in range(num_objects):
    val_input.append(val_x_images)
    val_input.append(val_x_features[:, i, :])

filepath = "./Saved_Models_and_Weights/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

history = model.fit(train_input, train_y, batch_size, num_epochs,
                    shuffle=True,
                    validation_data=(val_input, val_y), verbose=True,
                    callbacks=[checkpoint])

with open('./Saved_Models_and_Weights/ModelHistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)