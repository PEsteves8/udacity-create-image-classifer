from utils import load_checkpoint, process_image
import argparse
from PIL import Image
import torch
import json

parser = argparse.ArgumentParser(description='Image Classifier Predict')

parser.add_argument("options", nargs="*", default=["flowers/test/1/image_06743.jpg", "checkpoints/imageclassifiercheckpoint.pth"], help="The path to file and the checkpoint to load")
parser.add_argument('--top_k', dest="top_k", type=int, default="1")
parser.add_argument('--category_names', dest="category_names", default="cat_to_name.json", help="File storing the map to category names")
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')

args = parser.parse_args()

image = Image.open(args.options[0])
model, classes_list = load_checkpoint(args.options[1])
topk = args.top_k
device = torch.device("cuda" if args.gpu else "cpu")
with open(args.category_names, 'r') as f:
    category_names = json.load(f)

model.eval()
model.to(device)

proc_image = process_image(image)
proc_image.to(device)

with torch.no_grad():
    log_ps = model(proc_image)
        
ps = torch.exp(log_ps)
top_p, top_class = ps.topk(topk, dim=1)

probs = top_p.data.numpy()[0]
classes = top_class.data.numpy()[0]
print("classes")
print(classes)
classes_ids = [classes_list[id] for id in classes]
labels = [category_names[str(id)] for id in classes_ids]

for label, prob in zip(labels, probs):
    print("{} -> {:.3f} %".format(label, prob * 100))

          




