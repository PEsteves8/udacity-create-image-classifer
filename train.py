import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import os

parser = argparse.ArgumentParser(description='Image Classifier Train')

parser.add_argument("data_directory", nargs="*", default=["flowers"], help="A directory containing a train and test folders with data for the nn")
parser.add_argument('--save-dir', dest="save_dir", default="checkpoints", help="The folder that will store the checkpoints")
parser.add_argument('--arch', dest="arch", default="vgg16", help="Any architecture available in torchvision models")
parser.add_argument('--learning_rate', dest="learning_rate", type=float, default="0.001")
parser.add_argument('--hidden_units', type=int, default="4096")
parser.add_argument('--epochs', dest="epochs", type=int, default="5")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help="If set, the gpu will be used instead of the cpu")

args = parser.parse_args()

data_dir = args.data_directory[0]

data_transforms = {
    "train": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
}

image_datasets = {
    "train": datasets.ImageFolder(data_dir + '/train', transform=data_transforms["train"]),
    "test": datasets.ImageFolder(data_dir + '/test', transform=data_transforms["test"]),
}

dataloaders = {
    "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
    "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64)
}

model = getattr(models, args.arch)
model = model(pretrained=True)
device = torch.device("cuda" if args.gpu else "cpu")

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(25088, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, 102),
            nn.LogSoftmax(dim=1))

model.classifier = classifier
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
epochs = int(args.epochs)

train_losses, test_losses = [], []

for e in range(epochs):
    print("Starting epoch {}".format(e + 1))
    running_loss = 0
    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        log_ps = model(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        model.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for inputs, labels in dataloaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model(inputs)
                test_loss += criterion(log_ps, labels).item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        train_losses.append(running_loss/len(dataloaders["train"]))
        test_losses.append(test_loss/len(dataloaders["test"]))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders["train"])),
              "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders["test"])),
              "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders["test"])))
        model.train()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layer_size': args.hidden_units,
              'learning_rate': args.learning_rate,
              'dropout': 0.2,
              'epochs': args.epochs,
              'state_dict': model.state_dict(),
              'classes_list': image_datasets['train'].classes
             }

torch.save(checkpoint, args.save_dir + '/imageclassifiercheckpoint.pth')