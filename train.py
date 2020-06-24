import numpy as np
import os as os
import imageio
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.autograd import Variable
from math import ceil

import model

device = torch.device('cuda')
num_epochs = 100

# Data Loading
dataset_path = './Generated_dataset/Task-2/'

train_x = []
train_y = []

for subtask in os.listdir(dataset_path):
    if subtask.startswith('task'):
        continue
    print(subtask)
    subtask_folder = os.path.join(dataset_path, subtask)

    task_no = subtask.split('-')[1]
    subtask_no = subtask.split('-')[2]

    file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
    data = np.load(os.path.join(dataset_path, file))
    print(data.shape)

    for seq in range(len(os.listdir(subtask_folder)) - 1):
        img_path = os.path.join(subtask_folder, str(seq) + '.png')
        image = imageio.imread(img_path).astype(np.float)
        image /= 255.0
        # plt.imshow(image)
        # plt.show()
        # print(image.shape)

        train_x.append(image)

        # interested in data of only last object (added object)
        cx = data[seq + 1][-1][0]
        cy = 1. - data[seq + 1][-1][1]

        train_y.append([cx, cy])

        seq += 1

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

train_x = train_x.reshape(train_x.shape[0], 3, 256, 256)
train_x = torch.from_numpy(train_x)

train_y = torch.from_numpy(train_y)

val_x = val_x.reshape(val_x.shape[0], 3, 256, 256)
val_x = torch.from_numpy(val_x)

val_y = torch.from_numpy(val_y)

model = model.ObjLearner(
    embedding_dim=2,
    hidden_dim=512,
    input_dims=(3, 256, 256),
    num_objects=3)

optimizer = Adam(model.parameters(), lr=0.07)
model.cuda()

print(model)

criterion = torch.nn.MSELoss().cuda()
batch_size = 8

train_x_batches = np.array_split(train_x, ceil(train_x.shape[0] / 8))
train_y_batches = np.array_split(train_y, ceil(train_y.shape[0] / 8))

for epoch in range(num_epochs):
    model.train()

    x_val, y_val = Variable(val_x), Variable(val_y)
    x_val = x_val.cuda().float().cuda()
    y_val = y_val.cuda().float().cuda()

    train_loss = 0

    for train_x, train_y in zip(train_x_batches, train_y_batches):
        x_train, y_train = Variable(train_x), Variable(train_y)
        x_train = x_train.cuda().float().cuda()
        y_train = y_train.cuda().float().cuda()

        optimizer.zero_grad()

        train_op = model.run(x_train)
        loss = criterion(train_op, y_train)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    val_op = model.run(x_val)
    loss = criterion(val_op, y_val)
    val_loss = loss.item()

    if epoch % 2 == 0:
        print('Epoch : ', epoch + 1, '\t', 'train loss :', train_loss, ' ', 'val loss :', val_loss)
