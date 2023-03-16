# PROGRAMMER: Ujjwal Singh
# DATE CREATED: 25th Feb 2023                                  
# REVISED DATE: 
# PURPOSE:  function loads and preprocesses the image data for training, validation, and testing.


import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
from torch.autograd import Variable

arch = {"vgg16":4096,"densenet121":1024}

def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms_training = transforms.Compose([transforms.Resize(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(35),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    data_transforms_testing = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image_datasets_training = datasets.ImageFolder(train_dir,data_transforms_training)
    image_datasets_validation = datasets.ImageFolder(valid_dir,data_transforms_testing)
    image_datasets_testing = datasets.ImageFolder(test_dir,data_transforms_testing)

    train_dataloaders = torch.utils.data.DataLoader(image_datasets_training,batch_size=64,shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(image_datasets_validation,batch_size=64,shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(image_datasets_testing,batch_size=64,shuffle=True)

    return train_dataloaders, valid_dataloaders, test_dataloaders, image_datasets_training