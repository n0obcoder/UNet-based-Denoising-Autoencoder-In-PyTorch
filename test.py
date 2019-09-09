import os, shutil, cv2
import numpy as np
import torch
from torchvision import transforms
from unet import UNet
from datasets import custom_test_dataset
import config as cfg

res_dir = cfg.res_dir

if os.path.exists(res_dir):
    shutil.rmtree(res_dir)

if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

test_dir = cfg.test_dir
test_dataset       = custom_test_dataset(test_dir, transform = transform)
test_loader        = torch.utils.data.DataLoader(test_dataset, batch_size = cfg.test_bs, shuffle = not True)

print('\nlen(test_dataset) : ', len(test_dataset))
print('len(test_loader)  : {}  @bs={}'.format(len(test_loader), cfg.test_bs))

# defining the model
model = UNet(n_classes = 1, depth = 3, padding = True).to(device)

ckpt_path = os.path.join(cfg.models_dir, cfg.ckpt)
ckpt = torch.load(ckpt_path)
print(f'\nckpt loaded: {ckpt_path}')
model_state_dict = ckpt['model_state_dict']
model.load_state_dict(model_state_dict)
model.to(device)

def get_img_strip(tensr):
    # shape: [bs,1,h,w]
    bs, _ , h, w = tensr.shape
    tensr2np = (tensr.cpu().numpy().clip(0,1)*255).astype(np.uint8)    
    canvas = np.ones((h, w*bs), dtype = np.uint8)
    for i in range(tensr.shape[0]):
        patch_to_paste = tensr2np[i, 0, :, :]
        canvas[:, i*w: (i+1)*w] = patch_to_paste
    return canvas

def denoise(noisy_imgs, out):
    noisy_imgs = get_img_strip(noisy_imgs)
    out = get_img_strip(out)
    denoised = np.concatenate((noisy_imgs, out), axis = 0)
    return denoised

print('\nDenoising noisy images...')
model.eval()
with torch.no_grad():
    for batch_idx, noisy_imgs in enumerate(test_loader):
        print('batch: {}/{}'.format(str(batch_idx + 1).zfill(len(str(len(test_loader)))), len(test_loader)), end='\r')
        noisy_imgs = noisy_imgs.to(device)
        out = model(noisy_imgs)
        denoised = denoise(noisy_imgs, out)
        denoised =  denoised
        cv2.imwrite(os.path.join(res_dir, f'denoised{str(batch_idx).zfill(3)}.jpg'), denoised)
        
print('\n\nresults saved in \'{}\' directory'.format(res_dir))
