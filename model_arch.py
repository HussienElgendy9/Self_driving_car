import torch
import torch.nn as nn

class CNNs(nn.Module):
    def __init__(self):
        super(CNNs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        
        num_features = 16 * 26 * 26  

        num_features += 1  

        self.fc1 = nn.Linear(in_features=num_features, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.15)

        self.fc_output = nn.Linear(in_features=64, out_features=2)

    def forward(self, X, speed):
        X = self.maxpool1(self.relu1(self.conv1(X)))
        X = self.maxpool2(self.relu2(self.conv2(X)))
        X = self.maxpool3(self.relu3(self.conv3(X)))
        
        X = X.view(X.size(0), -1) 
        speed = speed.view(-1, 1)
        X = torch.cat((X, speed), dim=1) 
        
        X = self.dropout1(self.relu4(self.fc1(X)))
        X = self.dropout2(self.relu5(self.fc2(X)))
        
        X = self.fc_output(X).view(-1, 2)
        
        return X
