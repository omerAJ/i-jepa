from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from transforms import make_transforms
import torch.nn as nn
import os
import sys
import torch
import copy
import torchvision

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
        self.head = nn.Linear(49152, output_dim)  

    def forward(self, x):
        x = self.encoder(x) 
        x = x.view(x.size(0), -1) 
        x = self.head(x) 
        return x


num_classes = 10  # Number of output classes, e.g., 10 for MNIST
model = EncoderWithLinearHead(encoder, num_classes).to(device)

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
pretrained_dict = checkpoint['encoder']
msg = model.load_state_dict(pretrained_dict)
crop_scale=(1.0, 1.0)
transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=False)
batch_size=4096
trainset = torchvision.datasets.MNIST(root=r'D:/omer/i-jepa/ijepa/src/datasets/mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=r'D:/omer/i-jepa/ijepa/src/datasets/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        all_preds.extend(preds.view_as(target).cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# Calculate Accuracy
accuracy = accuracy_score(all_targets, all_preds)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
conf_mat = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()