import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class InstaDataset(Dataset):
    """
    A dataset to load images from pandas metadata.
    
    Args:
        df_json_path (string): path to json file to initialize pandas dataframe.
        img_dir (string): path directory with images.
        transform (Transform): optional transform to be applied to images.
    """
    def __init__(self, 
                 df_json_path, 
                 img_dir, 
                 transform=None,
                 transform_target=True):
        
        # Don't forget to load image ids as strings to avoid problems.
        self.df = pd.read_json(df_json_path, dtype={'id': 'string'})
        self.df_json_path = df_json_path
        self.img_dir = img_dir
        self.transform = transform
        self.transform_target = transform_target
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path_to_image = self.img_dir + self.df.id[idx] + '.jpg'
        img = Image.open(path_to_image)
        if self.transform:
            img = self.transform(img)
            
        target = torch.Tensor([self.df.target[idx]])
        if self.transform_target:
            target = torch.exp(-7 * target)
            
        return img, target