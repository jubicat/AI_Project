import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

pth = '/Users/gultajseyid/Desktop/ai_project/asl_dataset/'
lst = os.listdir(pth)
needed_path = '/asl_dataset/'

path = []
tClass = []
for i in lst:
    for j in os.listdir(pth+i):
        path.append(needed_path+i+'/'+j)
        tClass.append(i)

d = {'pathname': path,'class': tClass}
df = pd.DataFrame(data=d)
df['mode'] = 'train'

df = df.sample(frac=1)
a = df.shape[0]
# 70% of dataset is train, 20% validation and 10% is test
a = int(a * 0.2)
df['mode'].iloc[:a] = 'val'
df['mode'].iloc[a:int(a*1.5)] = 'test'
df.to_csv('csv.csv')

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# print("Path to dataset files:", path)