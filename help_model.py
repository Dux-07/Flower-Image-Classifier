# PROGRAMMER: Ujjwal Singh
# DATE CREATED: 25th Feb 2023                                  
# REVISED DATE: 
# PURPOSE:  For training, saving and loading a neural network model for image classification, and also for processing and 
#           predicting on images using a pre-trained model.

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import load_data


def set_network(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='gpu'):
    ''' 
    The purpose of the "set_network" function is to set up a neural network model for image classification, with the specified structure,       number of hidden units, dropout rate, and learning rate. The function also sets up the optimizer and loss function for training the         model

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    for para in model.parameters():
        para.requires_grad = False   

    from collections import OrderedDict

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088,4096)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.5)),
                                ('fc2', nn.Linear(4096,102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
    criterion = nn.NLLLoss()
    print(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, criterion


def save_checkpoint(image_datasets_training, model = 0, path = 'checkpoint.pth', structure = 'vgg16', hidden_units = 4096, dropout = 0.3, lr = 0.001, epochs = 1):
    '''
    The save_checkpoint function saves the trained model and other relevant information such as the class to index mapping, number of           epochs, and optimizer state dictionary to a specified path.

    '''
    model.class_to_idx = image_datasets_training.class_to_idx
    state = {'input_size': 25088,
                'output_size': output_no,
                'structure': 'vgg16',
                'learning_rate': 0.001,
                'classifier': model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(state , 'checkpoint.pth')
    
def load_checkpoint(path = 'checkpoint.pth'):
    """
    The load_checkpoint function loads a previously saved checkpoint file and returns the corresponding model, optimizer, and other relevant     information.

    """
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']

    model, _ = set_network(structure, 0.5, 4096, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_change = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    pic = img_change(img_pil)
    
    return pic


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to('cuda')
    model.eval()
    pic = process_image(image_path).numpy()
    pic = torch.from_numpy(np.array([pic])).float()

    with torch.no_grad():
        logps = model.forward(pic.cuda())
        
    prob = torch.exp(logps).data
    
    return prob.topk(topk)