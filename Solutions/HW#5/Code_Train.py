# -*- coding: utf-8 -*-
"""05-674579894-Vassef.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OMCoLN79LLDjNb36uvV-bj_ALkZqKnUi
"""

import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from google.colab import drive
drive.mount('/content/drive/')

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
creating/Loading the newly-generated .zip dataset
This is just an illustration of how I created the new .zip file containing the train and test sets with 9 classes per each. If you are a first-time user, you will create this zip file in '/content/drive/MyDrive/UIC/organized_data' directory; otherwise, you will unzip the existed zip file and save it locally before feeding its data to DataLoader attribute.
"""

# Commented out IPython magic to ensure Python compatibility.
# List of classes
classes = ['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']
classes.sort()
# Define the paths
source_dir = '/content/dataset/output'  # Local directory containing all the PNG images
output_dir = '/content/dataset'  # Directory to store the organized dataset
# Create the train and test directories
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

Organized_zip_datafile = '/content/drive/MyDrive/UIC/organized_data'

if not os.path.exists(Organized_zip_datafile): # First time-user / Don't have the newly-generated .zip file !

    os.mkdir(Organized_zip_datafile)
    # Unzipping the uploaded .zip file (Unorganized one)
    zip_file_path = '/content/drive/MyDrive/UIC/geometry_dataset.zip' # The one that was uploaded in Piaza!

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    # Create class directories in train and test folders
    for class_name in classes:
        class_train_dir = os.path.join(train_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_test_dir, exist_ok=True)

    # Move the images to the respective class folders
    for class_name in classes:
        class_images = [img for img in os.listdir(source_dir) if img.startswith(class_name)]
        random.shuffle(class_images)  # Shuffle the images
        train_images = class_images[:8000]  # First 8000 images for training
        test_images = class_images[8000:]  # Remaining images for testing

        for img in train_images:
            source_path = os.path.join(source_dir, img)
            dest_path = os.path.join(train_dir, class_name, img)
            shutil.copy(source_path, dest_path)

        for img in test_images:
            source_path = os.path.join(source_dir, img)
            dest_path = os.path.join(test_dir, class_name, img)
            shutil.copy(source_path, dest_path)

    print("Dataset organized into train and test sets.")

#     %rm -rf dataset/output # Remove the output folder before zipping the organized data folder!
    !zip -r /content/drive/MyDrive/UIC/organized_data/dataset.zip /content/dataset
#     %rm -rf dataset # remove the local folder to free up space!

# Unzip the previously-saved .zip file from the google drive path!
with zipfile.ZipFile(os.path.join(Organized_zip_datafile, 'dataset.zip'), 'r') as zip_ref:
    zip_ref.extractall('')
# Create our transformation to apply to our data!
transform = transforms.Compose([
transforms.Resize((64, 64)),  # Downsampling to speed up processing
transforms.ToTensor(),
])
# Convert the images into tensor format using the datasets.ImageFolder class!
train_dataset = datasets.ImageFolder(root = os.path.join('/content' + output_dir, 'train'), transform = transform)
test_dataset = datasets.ImageFolder(root = os.path.join('/content'+ output_dir, 'test'), transform = transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

"""Let's check the sizes!"""

X,y = next(iter(train_loader))

print('Data shapes (train/test):')
print( X.data.shape )

# and the range of pixel intensity values
print('\nData value range:')
print( (torch.min(X.data),torch.max(X.data)) )

"""Let's inspect a few random images"""

fig,axs = plt.subplots(4,4,figsize=(10,10))

for (i,ax) in enumerate(axs.flatten()):

  # extract that image (need to transpose it back to 32x32x3)
  pic = X.data[i].numpy().transpose((1,2,0))
  pic = pic/2 + .5 # undo normalization

  # and its label
  label = train_dataset.classes[y[i]]

  # and show!
  ax.imshow(pic)
  ax.text(16,0,label,ha='center',fontweight='bold',color='k',backgroundcolor='y')
  ax.axis('off')

plt.tight_layout()
plt.show()

"""Model's structure !
* What I tried?
    * **Kernel size** of 3 vs 5, where kernel size of 3 worked a little bit better as reducing the kernel size will result in larger dimention for the first feedforward layer (Why! The features'dimention of the next layer in CNN is linked to the features'dimention of the previous layer as the following):

     **Next_layer_dim (n)** =
     floor((**Pre_layer_dim (m)** + 2 * padding - kernel size)/stride) + 1

    * The number of feedforwad layer: I first went with 2 fc layers, but the performance roughly reached to **70%** acc on testset, so I decided to add one fully-connected layer to increase the model's complexity. The results after adding this layer became 84% acc on testset.

    * The effect of **padding size** between 1 and 2 was not that pronounced, atleast in my design!
    * By increasing the stride per-layer, the model gets deeper and the receptive field increases, meaning that we can capture even more in-depth features of the input images, That is, arguably, our model is going the better learn the abstract of iamges, causing more robustness and generalization. It's worth mentioning that the maxpooling layer is functioning similary. In this problem, I set stride in each of the convolution layers equal to one for ease of dimentions' calculations!

* **Loss function** => CrossEntropyLoss
* **Optimizer** => Adam with weighting decay of 1e-5 and learning rate of 1e-3.
"""

def makeTheNet():
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.bnorm1 = nn.BatchNorm2d(32)
            # dim before pooling: res = floor((64 + 2 * 1 - 3)/1) + 1 = 64, dim after pooling: 64/2 = 32
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bnorm2 = nn.BatchNorm2d(64)
            # dim before pooling: res = floor((32 + 2 * 1 - 3)/1) + 1 = 32, dim after pooling: 64/2 = 16
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = nn.MaxPool2d(2, 2)(x)
            x = nn.ReLU()(self.bnorm1(x))

            x = self.conv2(x)
            x = nn.MaxPool2d(2, 2)(x)
            x = nn.ReLU()(self.bnorm2(x))


            x = x.view(-1, 64 * 16 * 16)
            x = self.fc1(x)
            x = nn.ReLU()(x)
            x = self.fc2(x)
            x = nn.ReLU()(x)
            x = self.fc3(x)
            return x

    # create the model instance
    net = CNN(9)

    # loss function
    lossfun = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5)

    return net,lossfun,optimizer

"""Create the instances of model, loss function and optimizer!"""

# test the model with one batch
net,lossfun,optimizer = makeTheNet()

X,y = next(iter(train_loader))
yHat = net(X)

# check size of output
print('\nOutput size:')
print(yHat.shape)

# now compute the loss
loss = lossfun(yHat,torch.squeeze(y))
print(' ')
print('Loss:')
print(loss)

"""Creating a function for training procedure!"""

# a function that trains the model

def funtion2trainTheModel():

    # number of epochs
    numepochs = 15

    # create a new model
    net,lossfun,optimizer = makeTheNet()

    # send the model to the GPU
    net.to(device)

    # initialize losses
    trainLoss = torch.zeros(numepochs)
    testLoss   = torch.zeros(numepochs)
    trainAcc  = torch.zeros(numepochs)
    testAcc    = torch.zeros(numepochs)


    # loop over epochs
    for epochi in range(numepochs):

        # loop over training data batches
        net.train() # switch to train mode
        batchLoss = []
        batchAcc  = []

        for X,y in train_loader:

            # push data to GPU
            X = X.to(device)
            y = y.to(device)

            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat,y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and accuracy from this batch
            batchLoss.append(loss.item())
            batchAcc.append( torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
            # end of batch loop...

        # and get average losses and accuracies across the batches
        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi]  = 100*np.mean(batchAcc)

        print(f'train loss for epoch {epochi} equals {trainLoss[epochi]}, and train acc equals {trainAcc[epochi]}')
        #### test performance (here done in batches!)
        net.eval() # switch to test mode
        batchAcc  = []
        batchLoss = []

        for X,y in test_loader:

            # push data to GPU
            X = X.to(device)
            y = y.to(device)

            # forward pass and loss
            with torch.no_grad():
                yHat = net(X)
                loss = lossfun(yHat,y)

            # loss and accuracy from this batch
            batchLoss.append(loss.item())
            batchAcc.append( torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
            # end of batch loop...

        # and get average losses and accuracies across the batches
        testLoss[epochi] = np.mean(batchLoss)
        testAcc[epochi]  = 100*np.mean(batchAcc)

        print(f'test loss for epoch {epochi} equals {testLoss[epochi]}, and test acc equals {testAcc[epochi]}\n')
    # end epochs

    # function output
    return trainLoss,testLoss,trainAcc,testAcc,net

"""Train!"""

# ~30 minutes with 15 epochs on GPU
trainLoss,testLoss,trainAcc,testAcc,net = funtion2trainTheModel()

"""Saving the model's last checkpoint for future analysis and inferences!"""

model_directory = '/content/drive/MyDrive/UIC/checkpoint' # model.pth'
if not os.path.exists(model_directory): # trying to not overwrite the checkpoint's directory!
    # Save the model's state dictionary to a file
    os.makedirs(model_directory, exist_ok = True)
    torch.save(net.state_dict(), os.path.join(model_directory, 'model.pth'))
    print("Model saved to model.pth")

"""Big picture!"""

fig,ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(trainLoss,'s-',label='Train')
ax[0].plot(testLoss,'s-',label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss (CEL)')
ax[0].set_title('Model loss')
ax[0].legend()

ax[1].plot(trainAcc,'s-',label='Train')
ax[1].plot(testAcc,'o-',label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
ax[1].legend()

plt.show()
