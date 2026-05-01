import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
from model import SimpleLeafNet

# --- 1. HARDWARE ACCELERATION ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🚀 Client running on: {device}")

# 2. Read which folder this phone is supposed to look at
if len(sys.argv) < 2:
    print("Error: Please provide data folder path. Example: python client.py data/client_abhishek")
    sys.exit(1)

data_folder = sys.argv[1] 
print(f"Loading images from: {data_folder}")

# --- 3. PYTORCH IMAGE LOADER (Supercharged for MobileNetV2) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # CRITICAL: MobileNetV2 requires these exact numbers to see colors correctly
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

dataset = ImageFolder(root=data_folder, transform=transform)

# --- 4. THE 15-CLASS MASTER MAP ---
GLOBAL_CLASSES = {
    'Pepper__bell___Bacterial_spot': 0,
    'Pepper__bell___healthy': 1,
    'Potato___Early_blight': 2,
    'Potato___Late_blight': 3,
    'Potato___healthy': 4,
    'Tomato_Bacterial_spot': 5,
    'Tomato_Early_blight': 6,
    'Tomato_Late_blight': 7,
    'Tomato_Leaf_Mold': 8,
    'Tomato_Septoria_leaf_spot': 9,
    'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
    'Tomato__Target_Spot': 11,
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 12,
    'Tomato__Tomato_mosaic_virus': 13,
    'Tomato_healthy': 14
}

new_samples = []
new_targets = []
for path, local_label in dataset.samples:
    class_name = dataset.classes[local_label] 
    if class_name in GLOBAL_CLASSES:
        global_label = GLOBAL_CLASSES[class_name] 
        new_samples.append((path, global_label))
        new_targets.append(global_label)

dataset.samples = new_samples
dataset.targets = new_targets

# High-Speed Windows Dataloader
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=True
)

# --- 5. LOAD THE AI BRAIN ---
model = SimpleLeafNet(num_classes=15).to(device)
criterion = nn.CrossEntropyLoss()
# Scalpel speed learning rate to protect the un-frozen layers
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- 6. THE FEDERATED LEARNING LOGIC ---
class AgriGuardClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"Training AI on {len(dataset)} local leaf images...")
        
        model.train()
        # Fast Federated Loop: 5 local epochs per round for DEEP learning
        for epoch in range(5):
            for images, labels in train_loader:
                # Send the batch of images to the fast hardware
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config={}), len(dataset), {}

print("Connecting to AgriGuard Server...")
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=AgriGuardClient())

# Save the final intelligent brain!
torch.save(model.state_dict(), "agriguard_model.pth")