# path to saving models
model_dir = 'models'

# path to saving loss plots
losses_dir = 'losses'

# path to the data directories
data_dir = 'data'
train_dir = 'train'
val_dir = 'val'
imgs_dir = 'imgs'
noisy_dir = 'noisy'
debug_dir = 'debug'

# text file to get text from
txt_file_dir = 'shitty_text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 2000
train_percentage = 0.8

resume = True      # False for trainig from scratch, True for loading a previously saved weight
ckpt='model02.pth' # model file path to load the weights from, only useful when resume is True
lr = 1e-5          # learning rate
epochs = 2         # epochs to train for 

# batch size for train and val loaders
batch_size = 16
