import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        self.fc1 = nn.Linear(32 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)

        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x