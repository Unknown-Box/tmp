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
# input_image = Image.open("tree.jpg")
# input_image = Image.open("Unknown.png")
prompt_image = Image.open("wood.jpg")
# prompt_image = Image.open("Unknown-2.png")
# prompt_image = Image.open("spoon.jpg")

# normalize image
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), 
    transforms.Resize((352, 352))
    # transforms.Resize((160, 160))
])
img = transform(input_image).unsqueeze(0)
pimg = transform(prompt_image).unsqueeze(0)
print(prompt_image.size)
print(pimg.shape)


# set prompt
prompts = ["something to fill"]
# prompts = sys.argv[1:]

import time
s = time.time()

# # prediction
with torch.no_grad():
    pred = model(img.repeat(1, 1, 1, 1), prompts)[0]
    # pred = model(img, pimg)[0]

e = time.time()

print(e-s)

# visualize prediction
_, axs = plt.subplots(1, 2, figsize=(9, 4))
for i, ax in enumerate(axs.flatten()):
    ax.axis("off")
    if i == 0:
        ax.text(0, -15, "input image")
        ax.imshow(input_image)
    else:
        masked = torch.sigmoid(pred[i-1][0])
        ax.text(0, -15, prompts[i-1])
        ax.imshow(masked)

# pred = torch.sigmoid(pred)
# print(pred.shape)
# print(pred.dtype, pred.min(), pred.max())

# plt.imshow(input_image)
# plt.imsave("out.png", torch.sigmoid(pred[0][0]))

_.savefig("out.png")