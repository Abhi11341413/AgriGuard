import torch.nn as nn
import torch.nn.functional as F

class SimpleLeafNet(nn.Module):
    def __init__(self):
        super(SimpleLeafNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
       
        self.fc1 = nn.Linear(16 * 112 * 112, 3) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112) # Flatten the image safely
        x = self.fc1(x)
        return x