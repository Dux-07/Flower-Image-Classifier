# PROGRAMMER: Ujjwal Singh
# DATE CREATED: 25th Feb 2023                                  
# REVISED DATE: 
# PURPOSE:  Train a neural network model and report its performance during training and validation.

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import json
from PIL import Image
import load_data
import help_model

parser = argparse.ArgumentParser(
    description = 'Parsers for train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/", help="This is directory for training the images")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth", help="This is dir where model will be saved after training")
parser.add_argument('--arch', action="store", default="vgg16", help ="Choose the CNN architecture for training and testing")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001,help="Here we define the learning rate for the training")
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512,help="Here we define hidden units and default is 512")
parser.add_argument('--epochs', action="store", default=3, type=int, help="This is the no. of epochs for traing model")
parser.add_argument('--dropout', action="store", type=float, default=0.2, help="Set the dropout probability for the fully connected layer in the classifier. ")
parser.add_argument('--gpu', action="store", default="gpu", help="If you wanna train you model on GPU using CUDA add this arg")

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
learning_rate = args.learning_rate
structure = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

if torch.cuda.is_available() and power == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def main():
    train_dataloaders, valid_dataloaders, test_dataloaders, image_datasets_training = load_data.load_data(where)
    model, criterion = help_model.set_network(structure,dropout,hidden_units,learning_rate,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    print("--Training starting--")
    epochs = 7
    print_every = 20
    run_loss = 0 
    run_accuracy = 0
    val_losses, train_losses = [],[]


    for epoch in range(epochs):
        batches = 0
        model.train()
        for images,labels in train_dataloaders:
            batches += 1
            images,labels = images.to(device),labels.to(device)
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1,dim=1)
            equal = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = equal.mean()
            optimizer.zero_grad()
            run_loss += loss.item()
            run_accuracy += accuracy.item()
            if batches%print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images,labels in valid_dataloaders:
                        images,labels = images.to(device),labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps,labels)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1,dim=1)
                        equal = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
                        accuracy = equal.mean()
                        validation_loss += loss.item()
                        validation_accuracy += accuracy.item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {run_loss/print_every:.3f}.. "
                  f'Running Training Accuracy: {run_accuracy/print_every*100:.3f}% '
                  f"Validation loss: {validation_loss/len(valid_dataloaders):.3f}.. "
                  f"Validation accuracy: {validation_accuracy/len(valid_dataloaders):.3f}")
            running_loss = 0
            model.train()
            
    
    model.class_to_idx = image_datasets_training.class_to_idx
    state = {'input_size': 25088,
                'output_size': 102,
                'structure': 'vgg16',
                'learning_rate': 0.001,
                'classifier': model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(state , 'checkpoint.pth')
if __name__ == "__main__":
    main()