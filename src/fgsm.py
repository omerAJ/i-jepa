import torch
import torch.nn as nn
import sys
import os
import copy
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from transforms import make_transforms
batch_size=128

# Add the parent directory to sys.path
script_dir = os.path.dirname(r"D:\omer\i-jepa\ijepa\src\linearProbing.py")  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
parent_dir2 = os.path.dirname(parent_dir)  # Get the parent directory
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir2)
from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)

device=torch.device('cuda:0')
patch_size=14
crop_size=224
pred_depth=12
pred_emb_dim=384
model_name='vit_tiny'
# model_name='vit_huge'
# checkpoint_path=r'D:\omer\i-jepa\ijepa\src\checkpoint_originalEncoder.pth.tar'
checkpoint_path=r'D:\omer\i-jepa\ijepa\src\checkpoint.pth.tar'

encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
target_encoder = copy.deepcopy(encoder)

class EncoderWithLinearHead(nn.Module):
    def __init__(self, encoder, output_dim):
        super(EncoderWithLinearHead, self).__init__()
        self.encoder = encoder 
        self.head = nn.Linear(49152, output_dim)  #49152, 327680

    def forward(self, x):
        x = self.encoder(x) 
        x = x.view(x.size(0), -1) 
        x = self.head(x) 
        return x


num_classes = 10  
model = EncoderWithLinearHead(encoder, num_classes).to(device)

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
pretrained_dict = checkpoint['encoder']
msg = model.load_state_dict(pretrained_dict)

device=torch.device('cuda:0')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# def getGradients(model, data, target):
#     output_index=0
#     x, target = data.to(device), target.to(device)
#     x.requires_grad = True
#     optimizer.zero_grad()
#     out = model(x)   
#     specific_output = out[:, output_index] 
#     loss = loss_fn(specific_output, target.float())   
#     loss.backward()

#     return x.grad.data

def getGradients(model, data, target):
    x, target = data.to(device), target.to(device)
    x.requires_grad = True
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    return x.grad.data

# grad = getGradients(model, data[0], data[1])
# print(grad.shape)

import numpy as np

def fgsm_data(model, x, target, epsilon=0.1):
    if len(x.shape) == 3:
        x = x.unsqueeze(dim=0)
        target = target.unsqueeze(dim=0)
    grads = getGradients(model, x, target)
    signs = np.sign(grads.cpu())
    # print(signs.shape, x.shape)
    x = x.to('cpu') + np.multiply(epsilon, signs)
    return x

import cv2
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("img", (224, 224))
# def show_image(x):
#     print("showing image")
#     x = x.cpu().numpy()
#     x = x.squeeze()
#     x = np.transpose(x, (1, 2, 0))
#     cv2.imshow('img', x)
#     cv2.waitKey(0)
    # plt.imshow(x)
    # plt.title("MNIST Image")
    # plt.axis('off')  # Hide the axis
    # plt.show()

crop_scale=(1.0, 1.0)

transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=False)

testset = torchvision.datasets.MNIST(root=r'D:/omer/i-jepa/ijepa/src/datasets/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

def show_images_side_by_side(img1, img2):
    # Convert tensors to numpy arrays
    img1 = img1.cpu().numpy().squeeze()
    img2 = img2.cpu().numpy().squeeze()

    # Transpose the images from (C, H, W) to (H, W, C)
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the images in the subplots
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide the axis

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Perturbed Image')
    axes[1].axis('off')  # Hide the axis

    # Show the plot
    plt.show()
model.eval()
data = next(iter(testloader))
x = fgsm_data(model, data[0][1], data[1][1])
show_images_side_by_side(data[0][1], x)
# show_image(x)

# def fgsm_dataset(model, loader):
#     adversarial_dataset = []

#     for data, target in tqdm(loader):
#         data, target = data.to(device), target.to(device)
#         adversarial_data = fgsm_data(model, data, target)
#         adversarial_dataset.extend(zip(adversarial_data.cpu().detach(), target.cpu().detach()))

#     return adversarial_dataset


# adversarial_dataset = fgsm_dataset(model, testloader)
# # adversarial_dataset

# adversarial_loader = torch.utils.data.DataLoader(adversarial_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# model.eval()
# all_preds = []
# all_targets = []

# with torch.no_grad():
#     for data, target in adversarial_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         preds = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
#         all_preds.extend(preds.view_as(target).cpu().numpy())
#         all_targets.extend(target.cpu().numpy())

# # Calculate Accuracy
# accuracy = accuracy_score(all_targets, all_preds)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # Confusion Matrix
# conf_mat = confusion_matrix(all_targets, all_preds)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()