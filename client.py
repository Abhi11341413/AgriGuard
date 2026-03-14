import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
from model import SimpleLeafNet

# 1. Read which folder this phone is supposed to look at
data_folder = sys.argv[1] 
print(f"Loading images from: {data_folder}")

# 2. PyTorch Image Loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load whatever pictures are in the folder normally
dataset = ImageFolder(root=data_folder, transform=transform)


GLOBAL_CLASSES = {
    'tomato_early_blight': 0, 
    'potato_early_blight': 1, 
    'healthy': 2
}

new_samples = []
new_targets = []
for path, local_label in dataset.samples:
    class_name = dataset.classes[local_label] 
    global_label = GLOBAL_CLASSES[class_name] 
    new_samples.append((path, global_label))
    new_targets.append(global_label)

dataset.samples = new_samples
dataset.targets = new_targets

train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# 3. Load the AI Brain
model = SimpleLeafNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 4. The Federated Learning Logic
class AgriGuardClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Receive math from the server
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    # Train on the local leaf pictures
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training AI on local leaf images...")
        
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(dataset), {}

print("Connecting to AgriGuard Server...")
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=AgriGuardClient())
torch.save(model.state_dict(), "agriguard_model.pth")