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
    
        # padding the images
        # img = np.pad(img, ((96, 96), (0,0)), constant_values=(0,0))    
        # noisy_img = np.pad(noisy_img, ((96, 96), (0,0)), constant_values=(0,0))    
    
        if self.transform is not None:            
            img = self.transform(img)             
            noisy_img = self.transform(noisy_img)  
        
        return img, noisy_img

    def __len__(self):
        return len(self.imgs_data)
    
class custom_test_dataset(Dataset):
    def __init__(self, data_dir, transform = None, out_size = (64, 256)):
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

        # aspect ratio of the image required to be fed into the model (height/width)
        out_ar = self.out_size[0]/self.out_size[1]

        # aspect ratio of the image read
        ar = img.shape[0]/img.shape[1] # heigth/width
        
        # aspect ratio is the padding criteria here         
        if ar >= out_ar:
            # pad width
            pad = int(img.shape[0]/out_ar) - img.shape[1]
            pad1 = int(pad/2)
            pad2 = int(img.shape[0]/out_ar) - img.shape[1] - pad1
            img = np.pad(img, ((0,0),(pad1, pad2)), constant_values=(0,0))
            img = cv2.resize(img, (self.out_size[1], self.out_size[0]))
        else:
            # pad height
            pad = int(img.shape[1]*out_ar) - img.shape[0]
            pad1 = int(pad/2)
            pad2 = int(img.shape[1]*out_ar) - img.shape[0] - pad1
            img = np.pad(img, ((pad1, pad2), (0,0)), constant_values=(0,0))
            img = cv2.resize(img, (self.out_size[1], self.out_size[0]))
        
        if self.transform is not None:            
            img = self.transform(img)             
        return img

    def __len__(self):
        return len(self.imgs_data)
