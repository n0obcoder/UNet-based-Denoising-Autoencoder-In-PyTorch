import os

# path to saving models
models_dir = 'models'

# path to saving loss plots
losses_dir = 'losses'

# path to the data directories
data_dir = 'data'
train_dir = 'train'
val_dir = 'val'
imgs_dir = 'imgs'
noisy_dir = 'noisy'
debug_dir = 'debug'

# depth of UNet 
depth = 5 # try decreasing the depth value if there is a memory error

# text file to get text from
txt_file_dir = 'shitty_text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 18000
train_percentage = 0.8

resume = not True  # False for trainig from scratch, True for loading a previously saved weight
ckpt='model08.pth' # model file path to load the weights from, only useful when resume is True
lr = 3e-6          # learning rate
epochs = 3         # epochs to train for 

# batch size for train and val loaders
batch_size = 32 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 25

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = 'results'
test_bs = 64
