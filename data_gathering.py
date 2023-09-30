import torch
import torch.nn as nn
import torch.optim as optim
from model_arch import CNNs
import csv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

data = pd.read_csv('D:\\Games\\Sim\\Data\\First take\\driving_log.csv', usecols=[0, 3, 4, 5, 6])
center_image_paths = data.iloc[:, 0].values
values = data.iloc[:, 1:].values.astype(float)

num_conv_output_features = 16 * (224 // 2 // 2 // 2)

class Data_set(Dataset):
    def __init__(self, center_image_paths, values, transform=None):
        self.center_image_paths = center_image_paths
        self.values = values
        self.transform = transform

    def __len__(self):
        return len(self.center_image_paths)

    def __getitem__(self, idx):
        center_image_path = self.center_image_paths[idx]

        center_image = Image.open(center_image_path).convert('RGB') 

        
        if self.transform:
            center_image = self.transform(center_image)


        steering_angle = torch.tensor(self.values[idx, 0], dtype=torch.float32)
        throttle = torch.tensor(self.values[idx, 1], dtype=torch.float32)
        reverse = torch.tensor(self.values[idx, 2], dtype=torch.float32)
        speed = torch.tensor(self.values[idx, 3], dtype=torch.float32)

        speed = speed.unsqueeze(0)
        
        return center_image, steering_angle, throttle, reverse, speed


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset = Data_set(center_image_paths, values, transform=transform)


batch_size = 4  
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



model = CNNs()
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (images, steering_angle, throttle, reverse, speed) in enumerate(data_loader):

        outputs = model(images, speed)
        loss = criterion(outputs, torch.cat((steering_angle.unsqueeze(1), throttle.unsqueeze(1)), dim=1))  


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (batch_idx + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(data_loader), loss.item()))

torch.save(model.state_dict(), 'trained_model.pth')

