from keras.models import load_model, Model
from loss_functions import loss_func
from base_model import get_base_model
from keras.layers import Input, Concatenate

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
channels = 5
num_epochs = 7
batch_size = 16
input_feature_dims = 14
num_objects = 4


def get_model():
    base_model = get_base_model()

    inputs = []
    outputs = []

    for i in range(num_objects):
        input_image = Input((IMAGE_WIDTH, IMAGE_HEIGHT, channels))
        input_features = Input((input_feature_dims,))
        inputs.append(input_image)
        inputs.append(input_features)

    for i in range(0, 2 * num_objects, 2):
        output = base_model([inputs[i], inputs[i + 1]])
        outputs.append(output)

    concatenated_output = Concatenate()(outputs)

    model = Model(inputs=inputs, outputs=concatenated_output)
    model.compile(loss=loss_func,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
