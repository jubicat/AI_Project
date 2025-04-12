# Import required PyTorch libraries
import torch
import torch.nn as nn
import torchvision.models as models  # Pretrained computer vision models

# Define a custom VGG16 model by extending nn.Module
class vgg16(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Load a pretrained VGG16 model from torchvision
        self.vgg16 = models.vgg16(pretrained=True)

        # Replace the last fully connected (FC) layer in the classifier
        # Original: classifier[6] = nn.Linear(4096, 1000)
        # Here we reduce output to 1024 to add a custom classification head
        self.vgg16.classifier[6] = nn.Linear(4096, 1024)

        # Add a ReLU activation layer (in-place to save memory)
        self.relu = nn.ReLU(inplace=True)

        # Final FC layer to map to the desired number of output classes
        self.newfc = nn.Linear(1024, n_classes)

    def forward(self, image):
        # Forward pass through the modified VGG16 network
        vgg_pred = self.vgg16(image)        # Output from modified classifier[6]
        vgg_pred = self.relu(vgg_pred)      # Apply ReLU activation
        vgg_pred = self.newfc(vgg_pred)     # Final classification layer

        return vgg_pred
