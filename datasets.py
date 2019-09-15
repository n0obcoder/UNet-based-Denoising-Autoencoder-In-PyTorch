import numpy as np
import os, glob, cv2, sys
from torch.utils.data import Dataset

class DAE_dataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.imgs_data       = self.get_data(os.path.join(self.data_dir, 'imgs'))
        self.noisy_imgs_data = self.get_data(os.path.join(self.data_dir, 'noisy'))
        
    def get_data(self, data_path):
        data = []
        for img_path in glob.glob(data_path + os.sep + '*'):
            data.append(img_path)
        return data
    
    def __getitem__(self, index):  
        # read images in grayscale, then invert them
        img       = 255 - cv2.imread(self.imgs_data[index] ,0)
        noisy_img = 255 - cv2.imread(self.noisy_imgs_data[index] ,0)
    
        if self.transform is not None:            
            img = self.transform(img)             
            noisy_img = self.transform(noisy_img)  

        return img, noisy_img

    def __len__(self):
        return len(self.imgs_data)
    
class custom_test_dataset(Dataset):
    def __init__(self, data_dir, transform = None, out_size = (64, 256)):
        assert out_size[0] <= out_size[1], 'height/width of the output image shouldn\'t not be greater than 1'
        self.data_dir = data_dir
        self.transform = transform
        self.out_size = out_size
        self.imgs_data       = self.get_data(self.data_dir)

    def get_data(self, data_path):
        data = []
        for img_path in glob.glob(data_path + os.sep + '*'):
            data.append(img_path)
        return data
    
    def __getitem__(self, index):  
        # read images in grayscale, then invert them
        img       = 255 - cv2.imread(self.imgs_data[index] ,0)
                
        # check if img height exceeds out_size height
        if img.shape[0] > self.out_size[0]:
            resize_factor = self.out_size[0]/img.shape[0]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        # check if img width exceeds out_size width
        if img.shape[1] > self.out_size[1]:
            resize_factor = self.out_size[1]/img.shape[1]
            img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
       
        # add padding where required
        # pad height
        pad_height = self.out_size[0] - img.shape[0]
        pad_top = int(pad_height/2)
        pad_bottom = self.out_size[0] - img.shape[0] - pad_top
        # pad width
        pad_width = self.out_size[1] - img.shape[1]
        pad_left = int(pad_width/2)
        pad_right = self.out_size[1] - img.shape[1] - pad_left
        
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=(0,0))    
        
        if self.transform is not None:            
            img = self.transform(img)      
        
        return img

    def __len__(self):
        return len(self.imgs_data)
