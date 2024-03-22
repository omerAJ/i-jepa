import torch
import sys
import os
import copy
import torch.nn as nn

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
# model_name='vit_tiny'
model_name='vit_huge'
# load_path=r"D:\omer\i-jepa\ijepa\logs\jepa-latest.pth.tar"
load_path=r"D:\omer\i-jepa\ijepa\checkpoints\IN1K-vit.h.16-448px-300e.pth.tar"


encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
target_encoder = copy.deepcopy(encoder)

encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)

class EncoderWithLinearHead(nn.Module):
    def __init__(self, encoder, output_dim):
        super(EncoderWithLinearHead, self).__init__()
        self.encoder = encoder 
        self.head = nn.Linear(327680, output_dim)  

    def forward(self, x):
        x = self.encoder(x) 
        x = x.view(x.size(0), -1) 
        x = self.head(x) 
        return x


num_classes = 10  # Number of output classes, e.g., 10 for MNIST
model = EncoderWithLinearHead(encoder, num_classes).to(device)

# If you want to freeze the encoder weights
for param in model.encoder.parameters():
    param.requires_grad = False


print(encoder)

import torchvision
import sys
import os
from transforms import make_transforms

crop_scale=(1.0, 1.0)
transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=False)
batch_size=2
trainset = torchvision.datasets.MNIST(root=r'D:/omer/i-jepa/ijepa/src/datasets/mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=r'D:/omer/i-jepa/ijepa/src/datasets/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: ", device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 2

train_loss = []
test_loss = []
save_path = r'D:\omer\i-jepa\ijepa\src\checkpoint_originalEncoder.pth.tar'
for epoch in range(epochs):
    print(f"epoch {epoch}/{epochs}")
    model.train()
    for data in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        optimizer.zero_grad()
        out = model(data[0].to(device))
        loss = loss_fn(out, data[1].to(device))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    if epoch%2==0:
        model.eval()
        for data in tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} - Testing"):
            out = model(data[0].to(device))
            loss = loss_fn(out, data[1].to(device))
            test_loss.append(loss.item())
    save_dict = {
            'encoder': model.state_dict(),         
        }
    
    torch.save(save_dict, save_path)
        

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Epoch {epoch+1} Training Loss')
plt.legend()
plt.show()

