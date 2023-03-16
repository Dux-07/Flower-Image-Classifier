# PROGRAMMER: Ujjwal Singh
# DATE CREATED: 26th Feb 2023                                  
# REVISED DATE: 
# PURPOSE:  Predict the class (or classes) of an image using a trained deep learning model.

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

parser = argparse.ArgumentParser(description = 'Parsers for predict.py')

parser.add_argument('input', default='./flowers/test/1/image_06764.jpg', nargs='?', action="store", type = str,help="Path to the input image file. If no file path is provided, the default image will be used. Please make sure to provide the full path to the image file including the file name and extension.")
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/", help="Path to the directory containing the image data.")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str, help="Path to checkpoint file and if not specified than it will going to take default './checkpoint.pth' ")
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help="Return the top K most likely classes. The default value is 5.")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help="Path to the file that maps categories to real names.")
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu", help="If you wanna train your model using the GPU")

args = parser.parse_args()
image_path = args.input
output_no = args.top_k
device = args.gpu
json_name = args.category_names
checkpoint_path = args.checkpoint

def main():
    model=help_model.load_checkpoint(checkpoint_path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    prob = help_model.predict(image_path, model, output_no)
    probability = np.array(prob[0][0])
    label = [name[str(i + 1)] for i in np.array(prob[1][0])]
    
    i = 0
    while i < output_no:
        print(f"{label[i]} is  probability of {probability[i]:.3f} ")
        i += 1
    print("Prediction processes completed successfully.")

    
if __name__== "__main__":
    main()