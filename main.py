import sys

import torch
import requests

from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# from models.clipseg import ClipDensePredT
from models.clipseg import CLIPDensePredT


# load model
# model = ClipDensePredT(version="ViT-B/16", reduce_dim=64)
model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
model.eval()

model.load_state_dict(torch.load("weights/rd64-uni.pth", map_location=torch.device('cpu')), strict=False)


# load image
input_image = Image.open("example_image.jpg")

# normalize image
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), 
    transforms.Resize((352, 352))
])
img = transform(input_image).unsqueeze(0)


# set prompt
# prompts = ["red", "bin", "wood", "a jar"]
prompts = sys.argv[1:]

# prediction
with torch.no_grad():
    pred = model(img.repeat(4, 1, 1, 1), prompts)[0]

# visualize prediction
_, axs = plt.subplots(1, 5, figsize=(15, 4))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")
    if i == 0:
        ax.text(0, -15, "input image")
        ax.imshow(input_image)
    else:
        masked = torch.sigmoid(pred[i-1][0])
        ax.text(0, -15, prompts[i-1])
        ax.imshow(masked)

_.savefig("out.png")