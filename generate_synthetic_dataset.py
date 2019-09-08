import sys, os, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

def q(text = ''):
    print(f'>{text}<')
    sys.exit()

def degrade_quality(img):
    h, w = img.shape[0], img.shape[1]
    fx=np.random.randint(50,100)/100
    fy=np.random.randint(50,100)/100
    # print('fx, fy: ', fx, fy)
    small = cv2.resize(img, (0,0), fx = fx, fy = fy) 
    img = cv2.resize(small,(w,h))
    return img

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

img_train_dir = os.path.join(data_dir, 'train', 'imgs')
noisy_train_dir = os.path.join(data_dir, 'train', 'noisy')
debug_train_dir = os.path.join(data_dir, 'train', 'debug')

img_val_dir = os.path.join(data_dir, 'val', 'imgs')
noisy_val_dir = os.path.join(data_dir, 'val', 'noisy')
debug_val_dir = os.path.join(data_dir, 'val', 'debug')

dir_list = [img_train_dir, noisy_train_dir, debug_train_dir, img_val_dir, noisy_val_dir, debug_val_dir]
for dir_path in dir_list:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

f = open("shitty_text.txt", "r")
text = f.read()
f.close()
lines_list = str.split(text, '\n')
while '' in lines_list:
    lines_list.remove('')

lines_word_list = [str.split(line) for line in lines_list]
words_list = [planet for sublist in lines_word_list for planet in sublist] 

print('len(words_list): ', len(words_list))

# random font style
font_list = [cv2.FONT_HERSHEY_COMPLEX, 
cv2.FONT_HERSHEY_COMPLEX_SMALL,
cv2.FONT_HERSHEY_DUPLEX,
cv2.FONT_HERSHEY_PLAIN,
cv2.FONT_HERSHEY_SIMPLEX,
cv2.FONT_HERSHEY_TRIPLEX,
cv2.FONT_ITALIC] # cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cursive

h, w = 64*4, 256*4 # size of synthetic images

word_count = 0
num_imgs = 35000
train_num = int(num_imgs*0.8) # training percent
print('num_imgs : ', num_imgs)
print('train_num: ', train_num)

img_count = 1
word_start_x = 5
word_end_y = 5
for i in tqdm(range(num_imgs)):
    # try:
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

        # writing the text on the image
        cv2.putText(img, print_text.upper(), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        noisy_img = img.copy()
        #################################################
        # add horizontal line at the bottom of the text

        black_coords = np.where(noisy_img == fontColor)

        ymin = np.min(black_coords[0])
        ymax = np.max(black_coords[0])
        xmin = np.min(black_coords[1])
        xmax = np.max(black_coords[1])

        h_start_x = 0 #np.random.randint(0,xmin)
        h_end_x   = np.random.randint(int(img.shape[1]*0.8), img.shape[1])
        h_length = h_end_x - h_start_x + 1
        num_h_lines = np.random.randint(10,30)
        h_lines = []
        h_start_temp = h_start_x
        # print('num_h_lines: ', num_h_lines)

        next_line = True
        num_line = 0
        while (next_line) and (num_line < num_h_lines):
            if h_start_temp < h_end_x:
                # print('h_start_temp: ', h_start_temp)
                h_end_temp = np.random.randint(h_start_temp + 1, h_end_x + 1)
                # print('h_end_temp  : ', h_end_temp)
                if h_end_temp < h_end_x:
                    # print('0')
                    h_lines.append([h_start_temp, h_end_temp]) 
                    h_start_temp = h_end_temp + 1
                    num_line += 1
                else:
                    h_lines.append([h_start_temp, h_end_x]) 
                    num_line += 1
                    # print('1')
                    next_line = False
            else:
                # print('2')
                next_line = False

        for h_line in h_lines:
            # print('h_line: ', h_line)
            col = np.random.choice(['black', 'white'], p = [0.65, 0.35])
            if col == 'black':
                # cv2.line(noisy_img, (h_line[0], ymax), (h_line[1], ymax), np.random.randint(0, 30), np.random.randint(3, 5))        
                

                x_points = list(range(h_line[0], h_line[1] + 1))
                x_points_black_prob = np.random.choice([0,1], size = len(x_points), p = [0.2, 0.8])
                
                for idx, x in enumerate(x_points):
                    if x_points_black_prob[idx]:
                        noisy_img[ ymax - np.random.randint(4): ymax + np.random.randint(4), x] = np.random.randint(0,30)  

                
                
                # print('---')
        #################################################
        # adding vertical lines
        vertical_bool = {'left': np.random.choice([0,1], p =[0.2, 0.8]), 'right': np.random.choice([0,1])} # [1 or 0, 1 or 0] whether to make vertical left line on left and right side of the image
        for left_right, bool_ in vertical_bool.items():
            # print(left_right, bool_)
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

        debug_img = np.ones((2*h, w), dtype = np.uint8)*255
        debug_img[0:h, :] = img
        debug_img[h:2*h, :] = noisy_img
        cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)

        img       = cv2.resize(img,       (0,0), fx = 0.25, fy = 0.25)
        noisy_img = cv2.resize(noisy_img, (0,0), fx = 0.25, fy = 0.25)
        debug_img = cv2.resize(debug_img, (0,0), fx = 0.25, fy = 0.25)

        if img_count <= train_num:            
            cv2.imwrite(os.path.join(data_dir, 'train', 'imgs', '{}.jpg'.format(str(img_count).zfill(6))), img) 
            cv2.imwrite(os.path.join(data_dir, 'train', 'noisy', '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
            cv2.imwrite(os.path.join(data_dir, 'train', 'debug', '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 
        else:
            cv2.imwrite(os.path.join(data_dir, 'val', 'imgs', '{}.jpg'.format(str(img_count).zfill(6))), img) 
            cv2.imwrite(os.path.join(data_dir, 'val', 'noisy', '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
            cv2.imwrite(os.path.join(data_dir, 'val', 'debug', '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 

        img_count += 1

    # except:
        # pass
