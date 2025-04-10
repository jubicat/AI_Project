from torchmetrics import F1Score
from torch.utils.data import DataLoader, Dataset
from model.resnet import ResNetModel
from model.vgg import vgg16
from torch.optim import SGD, Adam
from tqdm.auto import tqdm
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
from timm import create_model
from torchmetrics.classification import BinaryAccuracy, Accuracy
tqdm.pandas()
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import os

save_model_path = "checkpoints/"
pth_name = "saved_model.pth"

dataset = pd.read_csv('/Users/gultajseyid/Desktop/ai_project/asl_alphabet_project/dataset/csv.csv')

train_df = dataset[dataset['mode'] == 'train'].reset_index()
val_df = dataset[dataset['mode'] == 'val'].reset_index()

train_df['class1'] = train_df['class'].astype('category')
train_df['class_code'] = train_df['class1'].cat.codes

val_df['class1'] = val_df['class'].astype('category')
val_df['class_code'] = val_df['class1'].cat.codes

currency_class = dict(zip(train_df['class1'].cat.categories,range(len(train_df['class1'].cat.categories))))

train_df['class_code'] = train_df['class1'].progress_map(lambda x: currency_class[x])
val_df['class_code'] = val_df['class1'].progress_map(lambda x: currency_class[x])
print(currency_class)
print('Number of Classes: ',  len(set(dataset['class'].to_list())))
# sys.exit(1)



class Dataset:
    def __init__(self, df, size):
        self.imgs = df['pathname']
        self.cur_class = df['class_code']

        self.tfms = A.Compose([
            A.Resize(size, size), 
            A.Rotate(limit=15, p = 0.05),
            A.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        prepath = "/Users/gultajseyid/Desktop/ai_project"
        img = Image.open(prepath+self.imgs[i]).convert('RGB')
        img = np.array(img)
        img = self.tfms(image = img)
        img = img['image']
        cur_class = torch.tensor(self.cur_class[i], dtype=torch.long)
        return img, cur_class        
    
def val(model, data_val, loss_function, writer, epoch, device):
    model.eval()  # Ensure model is in evaluation mode
    f1 = F1Score(num_classes=36, task='multiclass').to(device)
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # No gradients needed
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
    f1score = f1.compute()  # Compute the final F1 score

    writer.add_scalar("Validation/F1", f1score, epoch)
    writer.add_scalar("Validation/Loss", avg_loss, epoch)

    tq.close()
    print(f"F1 Score: {f1score.item():.4f}, Average Loss: {avg_loss:.4f}")

    # Reset metric
    f1.reset()

    return avg_loss, f1score



def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device):
    writer = SummaryWriter()
    model.to(device)  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    
    for epoch in range(n_epochs): 
        model.train()
        running_loss = 0.0
        
        tq = tqdm(total=len(train_loader))
        tq.set_description('epoch %d' % (epoch))
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        
        writer.add_scalar("Training Loss", running_loss/len(train_loader), epoch)
           
        tq.close()
        epoch_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
                
        #check the performance of the model on unseen dataset4
        val(model, val_loader, loss_fn, writer, epoch, device)
        
        #save the model in pth format
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)




def main():
    device = "cpu"
    
    train_data = Dataset(train_df, 256) # 224 for ResNet18, 256 for VGG16
    val_data = Dataset(val_df, 256) #224 for ResNet18, 256 for VGG16
    
    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=2,
        drop_last=True
    )

    model = ResNetModel(36).to(device)   # will be VGG18Base and ResNet18Base seperately
    optimizer = Adam(model.parameters(), lr=5e-5) # will be 0.00005, 0.001 and 0.1, SGD will be tested
    loss = nn.CrossEntropyLoss()

    max_epoch = 10 # 10 for vgg16, 20 for resnet18


    train(model, train_loader, val_loader,  optimizer, loss, max_epoch, device)
    
    
if __name__ == "__main__":
    main()