# Importing necessary libraries
from torchmetrics import F1Score
from torch.utils.data import DataLoader, Dataset
from model.resnet import ResNetModel  # Custom ResNet model implementation
from model.vgg import vgg16  # Custom VGG16 model implementation
from torch.optim import SGD, Adam
from tqdm.auto import tqdm
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
from timm import create_model  # For creating pretrained models
from torchmetrics.classification import BinaryAccuracy, Accuracy
tqdm.pandas()  # Enables progress bars in pandas operations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import os

# Paths for saving the model
save_model_path = "checkpoints/"
pth_name = "ResNet18_model.pth"

# Reading the dataset CSV file
dataset = pd.read_csv('/Users/gultajseyid/Desktop/ai_project/asl_alphabet_project/dataset/data_labeled.csv')

# Splitting dataset into train and validation based on 'mode' column
train_df = dataset[dataset['mode'] == 'train'].reset_index()
val_df = dataset[dataset['mode'] == 'val'].reset_index()

# Encoding class labels as categorical codes
train_df['class1'] = train_df['class'].astype('category')
train_df['class_code'] = train_df['class1'].cat.codes

val_df['class1'] = val_df['class'].astype('category')
val_df['class_code'] = val_df['class1'].cat.codes

# Creating mapping from class name to code
gesture_class = dict(zip(train_df['class1'].cat.categories, range(len(train_df['class1'].cat.categories))))

# Mapping class labels to codes using tqdm for progress visualization
train_df['class_code'] = train_df['class1'].progress_map(lambda x: gesture_class[x])
val_df['class_code'] = val_df['class1'].progress_map(lambda x: gesture_class[x])

print(gesture_class)
print('Number of Classes: ', len(set(dataset['class'].to_list())))


# Custom Dataset class for loading and transforming images
class Dataset:
    def __init__(self, df, size):
        self.imgs = df['pathname']
        self.ges_class = df['class_code']

        # Albumentations transformations for data augmentation and normalization
        self.tfms = A.Compose([
            A.Resize(size, size), 
            A.Rotate(limit=15, p=0.05),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        prepath = "/Users/gultajseyid/Desktop/ai_project/Project"
        img = Image.open(prepath + self.imgs[i]).convert('RGB')  # Load and convert image to RGB
        img = np.array(img)  # Convert to NumPy array for Albumentations
        img = self.tfms(image=img)['image']  # Apply transformations
        ges_class = torch.tensor(self.ges_class[i], dtype=torch.long)  # Convert label to tensor
        return img, ges_class


# Validation function to evaluate model on validation data
def val(model, data_val, loss_function, writer, epoch, device):
    model.eval()  # Set model to evaluation mode
    f1 = F1Score(num_classes=36, task='multiclass').to(device)  # Metric for evaluation
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        tq = tqdm(total=len(data_val), desc='Validation', leave=False)
        for batch in data_val:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = loss_function(preds, labels)
            total_loss += loss.item()

            preds_softmax = torch.softmax(preds, dim=1)
            preds_labels = torch.argmax(preds_softmax, dim=1)
            f1.update(preds_labels, labels)

            total_samples += labels.size(0)
            tq.update()

    avg_loss = total_loss / total_samples
    f1score = f1.compute()

    # Log metrics to TensorBoard
    writer.add_scalar("Validation/F1", f1score, epoch)
    writer.add_scalar("Validation/Loss", avg_loss, epoch)

    tq.close()
    print(f"F1 Score: {f1score.item():.4f}, Average Loss: {avg_loss:.4f}")

    f1.reset()  # Reset the metric state
    return avg_loss, f1score


# Training loop
def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device):
    writer = SummaryWriter()  # For TensorBoard logging
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        tq = tqdm(total=len(train_loader))
        tq.set_description('epoch %d' % (epoch))

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # Reset gradients

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            outputs = outputs.softmax(dim=1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)

        # Log training loss to TensorBoard
        writer.add_scalar("Training Loss", running_loss / len(train_loader), epoch)

        tq.close()
        epoch_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, n_epochs, epoch_loss))

        # Evaluate on validation set
        val(model, val_loader, loss_fn, writer, epoch, device)

        # Save model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)


# Main function to initialize and run training
def main():
    device = "cpu"  

    # Image size: 224 for ResNet18, 256 for VGG16
    train_data = Dataset(train_df, 256)
    val_data = Dataset(val_df, 256)

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2, drop_last=True)

    # Initialize model, optimizer, and loss function
    model = ResNetModel(36).to(device)  # 36 classes
    optimizer = Adam(model.parameters(), lr=5e-5)  # Learning rate 0.00005
    loss = nn.CrossEntropyLoss()  # Suitable for multi-class classification

    max_epoch = 10  # Training for 10 epochs

    train(model, train_loader, val_loader, optimizer, loss, max_epoch, device)


if __name__ == "__main__":
    main()
