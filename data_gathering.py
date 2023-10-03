import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import CNNs
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('D:\\Games\\Sim\\Data\\First take\\driving_log.csv', usecols=[0, 3, 4, 5])
center_image_paths = data.iloc[:, 0].values
values = data.iloc[:, 1:].values.astype(float)

class Data_set(Dataset):
    def __init__(self, center_image_paths, values):
        self.center_image_paths = center_image_paths
        self.values = values

    def __len__(self):
        return len(self.center_image_paths)

    def __getitem__(self, idx):
        center_image_path = self.center_image_paths[idx]
        
        center_image = Image.open(center_image_path).convert('RGB') 
        center_image_tensor=torchvision.transforms.functional.to_tensor(center_image)

        
        steering_angle=torch.tensor(self.values[idx ,0] ,dtype=torch.float32)
        throttle=torch.tensor(self.values[idx ,1] ,dtype=torch.float32)
        reverse=torch.tensor(self.values[idx ,2] ,dtype=torch.float32)

        
        return center_image_tensor ,steering_angle ,throttle ,reverse

batch_size = 4  
dataset = Data_set(center_image_paths ,values)
data_loader = DataLoader(dataset ,batch_size=batch_size ,shuffle=True)


model=CNNs().to(device)
criterion=nn.MSELoss()  
optimizer=optim.Adam(model.parameters() ,lr=0.001)

num_epochs=150
for epoch in range(num_epochs):
    for batch_idx ,(images ,steering_angle ,throttle ,reverse) in enumerate(data_loader):
        images=images.to(device)
        steering_angle=steering_angle.to(device)
        throttle=throttle.to(device)
        outputs=model(images)
        loss=criterion(outputs ,torch.cat((steering_angle.unsqueeze(1) ,throttle.unsqueeze(1)) ,dim=1))  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Epoch [{}/{}] ,Step [{}/{}] ,Loss: {:.4f}'.format(epoch+1 ,num_epochs ,batch_idx+1 ,len(data_loader) ,loss.item()))

torch.save(model ,'model.pth')
