import torch
from torchvision import datasets, transforms, models
import numpy as np
from torch import nn

def load_checkpoint(filepath):
    model = models.vgg16(pretrained=True)
    
    checkpoint = torch.load(filepath)
    
    classifier = nn.Sequential(
            nn.Dropout(checkpoint["dropout"]),
            nn.Linear(checkpoint["input_size"], checkpoint["hidden_layer_size"]),
            nn.ReLU(),
            nn.Dropout(checkpoint["dropout"]),
            nn.Linear(checkpoint["hidden_layer_size"], checkpoint["output_size"]),
            nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    classes_list = checkpoint["classes_list"]
    
    return model, classes_list

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    trans = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    return trans(image).float().unsqueeze(0)
