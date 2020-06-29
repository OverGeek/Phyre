from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input, concatenate

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
channels = 5
batch_size = 16

num_classes = 100

input_feature_dims = 14  # 4(timesteps) x 3(vx, vy, theta) + 2(shape)


def get_base_model():
    input_image = Input((IMAGE_WIDTH, IMAGE_HEIGHT, channels))
    input_features = Input((input_feature_dims,))

    x = Conv2D(128, (10, 10), activation='relu')(input_image)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (8, 8), activation='relu', strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(512, (5, 5), activation='relu', strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(5, 5))(x)

    x = Flatten()(x)  # 512

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = concatenate([x, input_features])

    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    vx = Dense(num_classes, activation='softmax')(x)
    vy = Dense(num_classes, activation='softmax')(x)
    theta = Dense(num_classes, activation='softmax')(x)

    output = concatenate([vx, vy, theta])

    model = Model(inputs=[input_image, input_features], outputs=output)

    return model


print(get_base_model().summary())

# model.compile(loss=loss_fun_task2,
#               optimizer='adam',
#               metrics=['accuracy'])

# print(model.summary())
# keras.utils.plot_model(model, 'model.png')
#
# model.save('Model_Task2')
