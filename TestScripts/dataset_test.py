import os
import numpy as np
import torchvision as tv 
from PIL import Image
import torch
import cv2
from torch.utils.data import DataLoader


class DataGenTest(torch.utils.data.Dataset):
  def __init__(self, X):
        self.x1 = X[:, 0]
        self.x2 = X[:, 1]
        self.x3 = X[:, 2]
        self.x4 = X[:, 3]
        self.x5 = X[:, 4]

  def __len__(self):
        return (len(self.x1))

  def __getitem__(self, idx):
        batch_x1 = self.x1[idx:idx + 1]
        batch_x2 = self.x2[idx:idx + 1]
        batch_x3 = self.x3[idx:idx + 1]
        batch_x4 = self.x4[idx:idx + 1]
        batch_x5 = self.x5[idx:idx + 1]
        
        X = [(([(cv2.imread(file_name)) for file_name in batch_x1])), 
        (([(cv2.imread(file_name)) for file_name in batch_x2])),      
        (([(cv2.imread(file_name)) for file_name in batch_x3])),      
        (([(cv2.imread(file_name)) for file_name in batch_x4])),      
        (([(cv2.imread(file_name)) for file_name in batch_x5]))]      

        new_x = torch.cat([torch.tensor(np.transpose(np.float32(xs[0]/255.0), (2, 0, 1))) for xs in X], dim= 0)
        return new_x


