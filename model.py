# import torch
# import torch.nn as nn
# import torchvision.models as models

# class SimpleLeafNet(nn.Module):
#     def __init__(self, num_classes=15): # Updated for your 15 Kaggle folders!
#         super(SimpleLeafNet, self).__init__()
        
#         # 1. Load the pre-trained MobileNetV2 brain
#         self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
#         # 2. Freeze the early layers to save massive computational power
#         for param in self.base_model.parameters():
#             param.requires_grad = False
            
#         # 3. Replace the final classification head for our 15 specific farm diseases
#         in_features = self.base_model.classifier[1].in_features
#         self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         return self.base_model(x)

import torch
import torch.nn as nn
import torchvision.models as models

class SimpleLeafNet(nn.Module):
    def __init__(self, num_classes=15):
        super(SimpleLeafNet, self).__init__()
        
        # 1. Load the pre-trained MobileNetV2 brain
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 2. FINE-TUNING: Freeze early layers, UNFREEZE the deep layers
        for name, param in self.base_model.named_parameters():
            if "features.18" in name or "classifier" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False 
                 
        # 3. Replace the final classification head for our 15 specific farm diseases
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)