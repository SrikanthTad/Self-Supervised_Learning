#!/usr/bin/env python
# coding: utf-8

# # SIMCLR
# 
# Take an input image
# Prepare 2 random augmentations on the image, including: rotations, hue/saturation/brightness changes, zooming, cropping, etc. 
# 
# Run a deep neural network (ResNet50) to obtain image representations (embeddings) for those augmented images.
# Run a small fully connected linear neural network to project embeddings into another vector space.
# Calculate contrastive loss and run backpropagation through both networks. 
# 
# Note: Contrastive loss decreases when projections coming from the same image are similar. The similarity between projections can be arbitrary, here I will use cosine similarity, same as in the paper.

# In[ ]:





# ## Contrastive loss:
# 
# Decreases when projections of augmented images coming from the same input image are similar.
# 
# For two augmented images: (i), (j) (coming from the same input image - I will call them “positive” pair later on), the contrastive loss for (i) tries to identify (j) among other images (“negative” examples) that are in the same batch .

# In[26]:



import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from torchvision import transforms as T
import torchvision
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet50

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:





# In[44]:


device = torch.device('cuda:2')
device

root_folder = 'myData/imagenet-5-categories'


# In[45]:


#Transformations

def get_color_distortion(s=1.0):
    color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter =  torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray =  torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort =  torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def deprocess_and_show(img_tensor):
    return  torchvision.transforms.Compose([
             torchvision.transforms.Normalize((0, 0, 0), (2, 2, 2)),
             torchvision.transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
             torchvision.transforms.ToPILImage()
          ])(img_tensor)


# In[46]:


#nt-xt loss

t= 0.5 #seems to be norm

def nxtLossFunction(in_1, in_2):
    in_1_norm = torch.norm(in_1, dim=1).reshape(-1, 1)
    in_1_div = torch.div(in_1, in_1_norm)
    in_2_norm = torch.norm(in_2, dim=1).reshape(-1, 1)
    in_2_div = torch.div(in_2, in_2_norm)
    
    concatTensor12 = torch.cat([in_1_div, in_2_div], dim=0)
    concatTensor_transpose = torch.t(concatTensor12)
    concatTensor21 = torch.cat([in_2_div, in_1_div], dim=0)
    
    sim = torch.mm(concatTensor12,concatTensor_transpose )
    sim_by_tau = torch.div(sim, t)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(concatTensor12, concatTensor21), t))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    loss = torch.div(numerators, denominators)
    neglog_loss = -torch.log(loss)
    return torch.mean(neglog_loss)


# In[47]:


# model = nn.Sequential(OrderedDict([
#           ('conv1', nn.Conv2d(1,20,5)),
#           ('relu1', nn.ReLU()),
#           ('conv2', nn.Conv2d(20,64,5)),
#           ('relu2', nn.ReLU())
#         ]))
from torchvision.models import resnet50
resnet = resnet50(pretrained=False)

fc_classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(resnet.fc.in_features, 100)),
    ('added_relu1', nn.ReLU(inplace=True)),
    ('fc2', nn.Linear(100, 50)),
    ('added_relu2', nn.ReLU(inplace=True)),
    ('fc3', nn.Linear(50, 25))
]))

resnet.fc = fc_classifier
resnet.to(device)


# In[48]:


class MyDataset(Dataset):
    def __init__(self, root_dir, filenames, labels, mutation=False):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels
        self.mutation = mutation

    def __len__(self):
        return len(self.file_names)

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
        return res

    def mutate_image(self, img):
        res = T.RandomResizedCrop(224)(img)
        res = get_color_distortion(1)(res)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        image = T.Resize((250, 250))(image)

        if self.mutation:
            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}
        else:
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)
            sample = {'image': image, 'label': label}

        return sample



# In[49]:


# import zipfile
# with zipfile.ZipFile("./myData/imagenet-5-categories-master.zip", 'r') as zip_ref:
#     zip_ref.extractall("./myData/imagenet-5-categories")

import os
import random

train_names = sorted(os.listdir(root_folder + '/train'))
test_names = sorted(os.listdir(root_folder + '/test'))

# setting random seed to ensure the same 10% labelled data is used when training the linear classifier
random.seed(0)

names_train_10_percent = random.sample(train_names, len(train_names) // 10)
names_train = random.sample(train_names, len(train_names))
names_test = random.sample(test_names, len(test_names))


train_names = sorted(os.listdir(root_folder + '/train'))
test_names = sorted(os.listdir(root_folder + '/test'))

# defining a mapping between class names and numbers
mapping = {'car': 0, 'dog': 1, 'elephant': 2, 'cat': 3, 'airplane': 4}
inverse_mapping = ['car', 'dog', 'elephant', 'cat', 'airplane']

# getting labels based on filenames, note that the filenames themselves contain classnames
# also note that these labels won't be used to actually train the base model
# these are just for visualization purposes
labels_train = [mapping[x.split('_')[0]] for x in names_train]
labels_test = [mapping[x.split('_')[0]] for x in names_test]

# these 10 percent labels will be used for training the linear classifer
labels_train_10_percent = [mapping[x.split('_')[0]] for x in names_train_10_percent]


# In[50]:


# datasets
training_dataset_mutated = MyDataset(root_folder + '/train', names_train, labels_train, mutation=True)
training_dataset = MyDataset(root_folder + '/train', names_train_10_percent, labels_train_10_percent, mutation=False)
testing_dataset = MyDataset(root_folder + '/test', names_test, labels_test, mutation=False)

# dataloaders
dataloader_training_dataset_mutated = DataLoader(training_dataset_mutated, batch_size=50, shuffle=True, num_workers=2)
dataloader_training_dataset = DataLoader(training_dataset, batch_size=50, shuffle=True, num_workers=2)
dataloader_testing_dataset = DataLoader(testing_dataset, batch_size=50, shuffle=True, num_workers=2)


# In[51]:


#Train Resnet


# In[52]:


losses_train = []
num_epochs = 10
TRAINING = True
# using Adam optimizer
optimizer = Adam(resnet.parameters(), lr=0.001)


# In[57]:


if TRAINING:
    # get resnet in train mode
    resnet.train()

    # run a for loop for num_epochs
    for epoch in range(num_epochs):

        # a list to store losses for each epoch
        epoch_losses_train = []

        # run a for loop for each batch
        loop = enumerate(dataloader_training_dataset_mutated)
        for (_, sample_batched) in loop:
            
            # zero out grads
            optimizer.zero_grad()

            # retrieve x1 and x2 the two image batches
            x1 = sample_batched['image1']
            x2 = sample_batched['image2']

            # move them to the device
            x1 = x1.to(device)
            x2 = x2.to(device)

            # get their outputs
            y1 = resnet(x1)
            y2 = resnet(x2)

            # get loss value
            loss = nxtLossFunction(y1, y2)

            # perform backprop on loss value to get gradient values
            loss.backward()

            # run the optimizer
            optimizer.step()


# In[58]:


#append Linear


# In[59]:


LINEAR = True

class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(25, 5)

    def forward(self, x):
        x = self.fc1(x)
        return(x)

if LINEAR:

    # getting our linear classifier
    linear_classifier = LinearNet()

    # moving it to device
    linear_classifier.to(device)

    # using SGD as a linear optimizer
    linear_optimizer = Adam(linear_classifier.parameters(), lr=0.001)

    #number of epochs
    num_epochs_linear = 10

    # Boolean variable to control training of linear classifier
    LINEAR_TRAINING = True


# In[60]:


pred_accuracy1 = 0.0
total1 = 0.0
for epoch in range(10):

        if LINEAR_TRAINING:

            # run linear classifier in train mode
            linear_classifier.train()

            # for loop for running through each batch
            for (_, sample_batched) in enumerate(dataloader_training_dataset):

                # get x and y from the batch
                x = sample_batched['image']
                y_actual = sample_batched['label']

                # move them to the device
                x = x.to(device)
                y_actual  = y_actual.to(device)

                # get output from resnet architecture
                y_intermediate = resnet(x)

                # zero the grad values
                linear_optimizer.zero_grad()

                # run y_intermediate through the linear classifier
                y_predicted = linear_classifier(y_intermediate)

                # get the cross entropy loss value
                loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
                
                # perform backprop through the loss value
                loss.backward()

                # call the linear_optimizer step function
                linear_optimizer.step()
                
                # get predictions and actual values to cpu  
                pred = np.argmax(y_predicted.cpu().data, axis=1)
                actual = y_actual.cpu().data
                
                pred_accuracy1 += (actual == pred).sum().item()
                total1 += len(actual)
    
                test_acc = pred_accuracy1 / total1
                print(test_acc)


# In[61]:


# run a for loop through each batch

pred_accuracy = 0.0
total = 0.0

for (_, sample_batched) in enumerate(dataloader_testing_dataset):
    x = sample_batched['image']
    y_actual = sample_batched['label']

    x = x.to(device)
    y_actual  = y_actual.to(device)

    y_intermediate = resnet(x)

    y_predicted = linear_classifier(y_intermediate)
    loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
#             epoch_losses_test_linear.append(loss.data.item())

    pred = np.argmax(y_predicted.cpu().data, axis=1)
    actual = y_actual.cpu().data
    
    pred_accuracy += (actual == pred).sum().item()
    total += len(actual)
    
    test_acc = pred_accuracy / total
    print(test_acc)






