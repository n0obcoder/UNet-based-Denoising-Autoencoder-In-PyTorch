import os, glob, cv2
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
        img       = 255 - cv2.imread(self.imgs_data[index] ,0)
        noisy_img = 255 - cv2.imread(self.noisy_imgs_data[index] ,0)
       
        if self.transform is not None:            
            img = self.transform(img)             
            noisy_img = self.transform(noisy_img)  
        return img, noisy_img

    def __len__(self):
        return len(self.imgs_data)
