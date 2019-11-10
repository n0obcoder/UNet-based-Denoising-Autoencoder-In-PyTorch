# UNet-based-Denoising-Autoencoder-In-PyTorch
Cleaning printed text using Denoising Autoencoder based on UNet architecture in PyTorch

## Requirements
* torch >= 0.4    
* torchvision >= 0.2.2
* opencv-python    
* numpy >= 1.7.3       
* matplotlib       
* tqdm             

## Generating Synthetic Data
Set the number of total synthetic images to be generated **num_synthetic_imgs** and set the percentage of training data **train_percentage** in *config.py*
Then run
```
python generate_synthetic_dataset.py
```
It will generate the synthetic data in a directory named *data* (can be changed in the config.py) in the root dirctory.

## Training
Set the desired values of **lr**, **epochs** and **batch_size** in *config.py*
### Start Training
In *config.py*,
* set **resume** to False

```
python train.py
```
### Resume Training
In *config.py*,
* set **resume** to True and
* set **ckpt** to the path of the model to be loaded, i.e. ckpt = 'model02.pth'

```
python train.py
```

## Testing
In *config.py*,
* set **ckpt** to the path of the model to be loaded, i.e. ckpt = 'model02.pth'
* set **test_dir** to the path that contains the noisy images that you need to denoise ('data/val/noisy' by default) 
* set **test_bs** to the desired batch size for the test set (1 by default)
```
python test.py
```
Once the testing is done, the results will be saved in a directory named *results*

## Results (Noisy and Denoised Image Pairs)
<div class="row">
*  <div class="column">
    <img src='/results/res01.png' width='250' alt='res01.png' hspace='15'>
  </div>
* <div class="column">
    <img src='/results/res02.png' width='250' alt='res02.png' hspace='15'>
  </div>
  <div class="column">
    <img src='/results/res03.png' width='250' alt='res03.png' hspace='15'>
  </div>
    <div class="column">
    <img src='/results/res04.png' width='250' alt='res04.png' hspace='15'>
  </div>
    <div class="column">
    <img src='/results/res05.png' width='250' alt='res05.png' hspace='15'>
  </div>
</div>
