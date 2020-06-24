from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from loss_functions import loss_fun_task2
import keras

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
channels = 5
batch_size = 16

num_classes = 100

input_velocity_dims = 8  # 4 - x and y velocities (4x2=8)

input_image = Input((IMAGE_WIDTH, IMAGE_HEIGHT, channels))
input_vel = Input((input_velocity_dims,))

x = Conv2D(64, (5, 5), activation='relu')(input_image)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (5, 5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)  # <=1000

x = concatenate([x, input_vel])  # , input_vel_3, , input_vel_4; red_ball_ch_-1, green_ball_ch_-1

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
vx = Dense(num_classes, activation='softmax')(x)
vy = Dense(num_classes, activation='softmax')(x)

output = concatenate([vx, vy])

model = Model(inputs=[input_image, input_vel], outputs=output)

model.compile(loss=loss_fun_task2,
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
keras.utils.plot_model(model, 'model.png')

model.save('Model_Task2')
