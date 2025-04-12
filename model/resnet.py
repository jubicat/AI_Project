# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torchvision.models as models  # Contains popular pre-trained models like ResNet

# Define a custom model based on ResNet18
class ResNetModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Load the ResNet18 architecture without pretrained weights
        # Set pretrained=True if you want to use ImageNet-pretrained weights
        self.resnet18 = models.resnet18(pretrained=False)

        # Replace the final fully connected layer (original: 512 → 1000)
        # Here, map 512 → 1024 to allow a custom classification head
        self.resnet18.fc = nn.Linear(512, 1024)

        # Add ReLU activation to introduce non-linearity
        self.relu = nn.ReLU(inplace=True)

        # Final fully connected layer to output logits for n_classes
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, image):
        # Pass input through modified ResNet18 model
        pred = self.resnet18(image)  # Outputs a 1024-dimensional vector
        pred = self.relu(pred)       # Apply ReLU activation
        pred = self.fc(pred)         # Output logits for each class

        return pred
