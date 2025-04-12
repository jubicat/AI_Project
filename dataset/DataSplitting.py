import torch
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

# Define the base path to the ASL dataset
base_path = '/Users/gultajseyid/Desktop/ai_project/asl_dataset/'
folder_list = os.listdir(base_path)
relative_path_prefix = '/asl_dataset/'

image_paths = []
class_labels = []

# Loop through each class folder and collect image paths and their labels
for class_name in folder_list:
    for image_name in os.listdir(os.path.join(base_path, class_name)):
        image_paths.append(f"{relative_path_prefix}{class_name}/{image_name}")
        class_labels.append(class_name)

# Create a DataFrame with image paths and corresponding class labels
data = {'pathname': image_paths, 'class': class_labels}
df = pd.DataFrame(data)
df['mode'] = 'train'  # Default mode

# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Assign 20% for validation and 10% for testing
total_samples = df.shape[0]
val_count = int(total_samples * 0.2)
test_count = int(val_count * 0.5)

df.loc[:val_count - 1, 'mode'] = 'val'
df.loc[val_count:val_count + test_count - 1, 'mode'] = 'test'

# Save the labeled data to a CSV file
df.to_csv('data_labeled.csv', index=False)

print(df.shape)
