import sys, os, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import config as cfg

def q(text = ''):
    print(f'>{text}<')
    sys.exit()

def degrade_quality(img):
    '''
    This function takes in an image (color or grayscale), downsizes it to a
    randomly chosen size and then resizes it to the original size of the image,
    degrading the quality of the image in the process.
    '''
    h, w = img.shape[0], img.shape[1]
    fx=np.random.randint(50,100)/100
    fy=np.random.randint(50,100)/100
    # print('fx, fy: ', fx, fy)
    small = cv2.resize(img, (0,0), fx = fx, fy = fy) 
    img = cv2.resize(small,(w,h))
    return img

data_dir = cfg.data_dir
train_dir = cfg.train_dir
val_dir = cfg.val_dir

imgs_dir = cfg.imgs_dir
noisy_dir = cfg.noisy_dir
debug_dir = cfg.debug_dir

train_data_dir = os.path.join(data_dir, train_dir)
val_data_dir = os.path.join(data_dir, val_dir)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)

if not os.path.exists(val_data_dir):
    os.mkdir(val_data_dir)

img_train_dir = os.path.join(data_dir, train_dir, imgs_dir)
noisy_train_dir = os.path.join(data_dir, train_dir, noisy_dir)
debug_train_dir = os.path.join(data_dir, train_dir, debug_dir)

img_val_dir = os.path.join(data_dir, val_dir, imgs_dir)
noisy_val_dir = os.path.join(data_dir, val_dir, noisy_dir)
debug_val_dir = os.path.join(data_dir, val_dir, debug_dir)

dir_list = [img_train_dir, noisy_train_dir, debug_train_dir, img_val_dir, noisy_val_dir, debug_val_dir]
for dir_path in dir_list:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

f = open(cfg.txt_file_dir, encoding='utf-8', mode="r")
text = f.read()
f.close()
lines_list = str.split(text, '\n')
while '' in lines_list:
    lines_list.remove('')

lines_word_list = [str.split(line) for line in lines_list]
words_list = [words for sublist in lines_word_list for words in sublist] 

print('number of words in the txt file: ', len(words_list))

# list of all the font styles
font_list = [cv2.FONT_HERSHEY_COMPLEX, 
             cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_PLAIN,
             cv2.FONT_HERSHEY_SIMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_ITALIC] # cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cursive

# size of the synthetic images to be generated
syn_h, syn_w = 64, 256

# scale factor
scale_h, scale_w = 4, 4

# initial size of the image, scaled up by the factor of scale_h and scale_w
h, w = syn_h*scale_h, syn_w*scale_w 

word_count = 0
num_imgs = int(cfg.num_synthetic_imgs) # max number of synthetic images to be generated
train_num = int(num_imgs*cfg.train_percentage) # training percent
print('\nnum_imgs : ', num_imgs)
print('train_num: ', train_num)

img_count = 1
word_start_x = 5 # min space left on the left side of the printed text
word_end_y = 5   # min space left on the bottom side of the printed text

print('\nsynthesizing image data...')
for i in tqdm(range(num_imgs)):
    # make a blank image
    img = np.ones((h, w), dtype = np.uint8)*255

    # set random parameters
    font = font_list[np.random.randint(len(font_list))]
    bottomLeftCornerOfText = (np.random.randint(word_start_x, int(img.shape[1]/3)), np.random.randint(int(img.shape[0]/2), int(img.shape[0]) - word_end_y))
    fontColor              = np.random.randint(0,30)# (np.random.randint(0,30),np.random.randint(0,30),np.random.randint(0,30))
    fontScale              = np.random.randint(2200,3000)/1000
    lineType               = np.random.randint(1,3)

    # text to be printed on the blank image
    num_words = np.random.randint(1,8)
    print_text = ''
    for _ in range(num_words):
        print_text += str.split(words_list[word_count])[0] + ' '
        word_count += 1
    print_text = print_text[:-1] # to get rif of the last space

    # writing the text (in UPPERCASE) on the image
    cv2.putText(img, print_text.upper(), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    
    # adding noise (horizontal and vertical lines) on the image containing text
    noisy_img = img.copy()

    ### add horizontal line at the bottom of the text
    black_coords = np.where(noisy_img == fontColor)
    # finding the extremes of the printed text
    ymin = np.min(black_coords[0])
    ymax = np.max(black_coords[0])
    xmin = np.min(black_coords[1])
    xmax = np.max(black_coords[1])

    h_start_x = 0 #np.random.randint(0,xmin)                           # min x of the horizontal line
    h_end_x   = np.random.randint(int(img.shape[1]*0.8), img.shape[1]) # max x of the horizontal line
    h_length = h_end_x - h_start_x + 1
    num_h_lines = np.random.randint(10,30) # partitions to be made in the horizontal line (necessary to make it look like naturally broken lines)
    h_lines = []
    
    h_start_temp = h_start_x
    next_line = True
    num_line = 0
    while (next_line) and (num_line < num_h_lines):
        if h_start_temp < h_end_x:
            h_end_temp = np.random.randint(h_start_temp + 1, h_end_x + 1)
            if h_end_temp < h_end_x:
                h_lines.append([h_start_temp, h_end_temp]) 
                h_start_temp = h_end_temp + 1
                num_line += 1
            else:
                h_lines.append([h_start_temp, h_end_x]) 
                num_line += 1
                next_line = False
        else:
            next_line = False

    for h_line in h_lines:
        col = np.random.choice(['black', 'white'], p = [0.65, 0.35]) # probabilities of line segment being a solid one or a broken one
        if col == 'black':
            x_points = list(range(h_line[0], h_line[1] + 1))
            x_points_black_prob = np.random.choice([0,1], size = len(x_points), p = [0.2, 0.8])

            for idx, x in enumerate(x_points):
                if x_points_black_prob[idx]:
                    noisy_img[ ymax - np.random.randint(4): ymax + np.random.randint(4), x] = np.random.randint(0,30)  

    ### adding vertical lines
    vertical_bool = {'left': np.random.choice([0,1], p =[0.2, 0.8]), 'right': np.random.choice([0,1])} # [1 or 0, 1 or 0] whether to make vertical left line on left and right side of the image
    for left_right, bool_ in vertical_bool.items():
        if bool_:
            if left_right == 'left':
                v_start_x = np.random.randint(5, int(noisy_img.shape[1]*0.06))
            else:
                v_start_x = np.random.randint(int(noisy_img.shape[1]*0.95), noisy_img.shape[1] - 5)

            v_start_y = np.random.randint(0, int(noisy_img.shape[0]*0.06))
            v_end_y   = np.random.randint(int(noisy_img.shape[0]*0.95), noisy_img.shape[0])

            y_points = list(range(v_start_y, v_end_y + 1))
            y_points_black_prob = np.random.choice([0,1], size = len(y_points), p = [0.2, 0.8])

            for idx, y in enumerate(y_points):
                if y_points_black_prob[idx]:
                    noisy_img[y, v_start_x - np.random.randint(4): v_start_x + np.random.randint(4)] = np.random.randint(0,30)  

    # '''
    # erode the image
    kernel = np.ones((3,3),np.uint8)
    erosion_iteration = np.random.randint(1,3)
    dilate_iteration = np.random.randint(0,2)
    img = cv2.erode(img,kernel,iterations = erosion_iteration)
    noisy_img = cv2.erode(noisy_img,kernel,iterations = erosion_iteration)
    img = cv2.dilate(img,kernel,iterations = dilate_iteration)
    noisy_img = cv2.dilate(noisy_img,kernel,iterations = dilate_iteration)
    # '''
    
    img = degrade_quality(img)
    noisy_img = degrade_quality(noisy_img)

    debug_img = np.ones((2*h, w), dtype = np.uint8)*255 # to visualize the generated images (clean and noisy)
    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = noisy_img
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)

    img       = cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)
    noisy_img = cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    debug_img = cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)

    if img_count <= train_num:            
        cv2.imwrite(os.path.join(data_dir, train_dir, imgs_dir, '{}.jpg'.format(str(img_count).zfill(6))), img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, noisy_dir, '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, debug_dir, '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 
    else:
        cv2.imwrite(os.path.join(data_dir, val_dir, imgs_dir, '{}.jpg'.format(str(img_count).zfill(6))), img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, noisy_dir, '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, debug_dir, '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 

    img_count += 1
