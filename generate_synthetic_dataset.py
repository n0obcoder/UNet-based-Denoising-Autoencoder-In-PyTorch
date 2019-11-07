import sys, os, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import shutil

import config as cfg

def q(text = ''):
    print(f'>{text}<')
    sys.exit()

data_dir = cfg.data_dir
train_dir = cfg.train_dir
val_dir = cfg.val_dir

imgs_dir = cfg.imgs_dir
noisy_dir = cfg.noisy_dir
debug_dir = cfg.debug_dir

train_data_dir = os.path.join(data_dir, train_dir)
val_data_dir = os.path.join(data_dir, val_dir)

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

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

def get_word_list():
    f = open(cfg.txt_file_dir, encoding='utf-8', mode="r")
    text = f.read()
    f.close()
    lines_list = str.split(text, '\n')
    while '' in lines_list:
        lines_list.remove('')

    lines_word_list = [str.split(line) for line in lines_list]
    words_list = [words for sublist in lines_word_list for words in sublist]

    return words_list


words_list = get_word_list()
print('\nnumber of words in the txt file: ', len(words_list))

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

img_count = 1
word_count = 0
num_imgs = int(cfg.num_synthetic_imgs) # max number of synthetic images to be generated
train_num = int(num_imgs*cfg.train_percentage) # training percent
print('\nnum_imgs : ', num_imgs)
print('train_num: ', train_num)


word_start_x = 5 # min space left on the left side of the printed text
word_start_y = 5
word_end_y = 5   # min space left on the bottom side of the printed text

def get_text():
    global word_count, words_list
    # text to be printed on the blank image
    num_words = np.random.randint(1,8)
    
    # renew the word list in case we run out of words 
    if (word_count + num_words) >= len(words_list):

        print('===\nrecycling the words_list')
        
        words_list = get_word_list() 
        word_count = 0

    print_text = ''
    for _ in range(num_words):
        print_text += str.split(words_list[word_count])[0] + ' '
        word_count += 1
    print_text = print_text.strip() # to get rif of the last space
    return print_text

def get_text_height(img, fontColor):
    black_coords = np.where(img == fontColor)
    # finding the extremes of the printed text
    ymin = np.min(black_coords[0])
    ymax = np.max(black_coords[0])
    # xmin = np.min(black_coords[1])
    # xmax = np.max(black_coords[1])
    ''' # for vizualising
    cv2.line(img, (0,ymin),(1000,ymin), 0,2)
    cv2.line(img, (0,ymax),(1000,ymax), 0,2)
    cv2.imshow('ymax', img)
    '''
    return ymax - ymin

def print_lines(img, font, bottomLeftCornerOfText, fontColor, fontScale, lineType, thickness):
    
    line_num = 0
    y_line_list = []

    # print('img.shape: ', img.shape)
    # print('initial bottomLeftCornerOfText: ', bottomLeftCornerOfText)

    while bottomLeftCornerOfText[1] <= img.shape[0]:
        # get a line of text
        print_text = get_text()

        # put it on a blank image and get its height
        if line_num == 0:
            # get the correct text height 
            big_img = np.ones((500,300), dtype = np.uint8)*255
            big_img_text = print_text.upper()
            cv2.putText(img = big_img, text = big_img_text, org = (0,200), fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
            text_height = get_text_height(big_img, fontColor)
            # print('text_height: ', text_height)

            if text_height > bottomLeftCornerOfText[1]:
                bottomLeftCornerOfText = (bottomLeftCornerOfText[0], np.random.randint(word_start_y, int(img.shape[0]*0.5)) + text_height)

            cv2.putText(img = img, text = print_text.upper(), org = bottomLeftCornerOfText, fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
            y_line_list.append(bottomLeftCornerOfText[1])
        else:
            # sampling the chances of adding one more line of text
            one_more_line = np.random.choice([0, 1], p = [0.4, 0.6])
            if not one_more_line:
                break
            cv2.putText(img = img, text = print_text.upper(), org = bottomLeftCornerOfText, fontFace = font, fontScale = fontScale, color = fontColor, thickness = thickness, lineType = lineType)
            y_line_list.append(bottomLeftCornerOfText[1])
        # calculate the (text_height+line break space) left on the bottom
        bottom_space_left = int(text_height*(1 + np.random.randint(20, 40)/100))
        # bottom_space_left = int(text_height*(1))
        # print('bottom_space_left: ', bottom_space_left)

        # update the bottomLeftCornerOfText
        bottomLeftCornerOfText = (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1] + bottom_space_left)

        # print('bottomLeftCornerOfText: ', bottomLeftCornerOfText)
        line_num += 1
    '''    
    for l in y_line_list:
        cv2.line(img, (0, l), (1000, l), 0, 1)
    '''
    return img, y_line_list, text_height

def get_noisy_img(img, y_line_list, text_height):

    # adding noise (horizontal and vertical lines) on the image containing text
    noisy_img = img.copy()
    
    # adding horizontal line (noise)
    for y_line in y_line_list: 

        # samples the possibility of adding a horizontal line
        add_horizontal_line = np.random.choice([0, 1], p = [0.5, 0.5])
        if not add_horizontal_line:
            continue

        # shift y_line randomly in the y-axis within a defined limit
        limit = int(text_height*0.3)
        if limit == 0: # this happens when the text used for getting the text height is '-', ',', '=' and other little symbols like these 
            limit = 10
        y_line += np.random.randint(-limit, limit)

        h_start_x = 0 #np.random.randint(0,xmin)                           # min x of the horizontal line
        h_end_x   = np.random.randint(int(noisy_img.shape[1]*0.8), noisy_img.shape[1]) # max x of the horizontal line
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
                        noisy_img[ y_line - np.random.randint(4): y_line + np.random.randint(4), x] = np.random.randint(0,30)  
    
    # adding vertical line (noise)
    vertical_bool = {'left': np.random.choice([0,1], p =[0.3, 0.7]), 'right': np.random.choice([0,1])} # [1 or 0, 1 or 0] whether to make vertical left line on left and right side of the image
    for left_right, bool_ in vertical_bool.items():
        if bool_:
            # print('left_right: ', left_right)
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

    return noisy_img

def degrade_qualities(img, noisy_img):
    '''
    This function takes in a couple of images (color or grayscale), downsizes it to a
    randomly chosen size and then resizes it to the original size,
    degrading the quality of the images in the process.
    '''
    h, w = img.shape[0], img.shape[1]
    fx=np.random.randint(50,100)/100
    fy=np.random.randint(50,100)/100
    # print('fx, fy: ', fx, fy)
    img_small = cv2.resize(img, (0,0), fx = fx, fy = fy) 
    img = cv2.resize(img_small,(w,h))
    
    noisy_img_small = cv2.resize(noisy_img, (0,0), fx = fx, fy = fy) 
    noisy_img = cv2.resize(noisy_img_small,(w,h))

    return img, noisy_img

def get_debug_image(img, noisy_img):
    debug_img = np.ones((2*h, w), dtype = np.uint8)*255 # to visualize the generated images (clean and noisy)
    debug_img[0:h, :] = img
    debug_img[h:2*h, :] = noisy_img
    cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 5)
    return debug_img

def erode_dilate(img, noisy_img):
    # erode the image
    kernel = np.ones((3,3),np.uint8)
    erosion_iteration = np.random.randint(1,3)
    dilate_iteration = np.random.randint(0,2)
    img = cv2.erode(img,kernel,iterations = erosion_iteration)
    noisy_img = cv2.erode(noisy_img,kernel,iterations = erosion_iteration)
    img = cv2.dilate(img,kernel,iterations = dilate_iteration)
    noisy_img = cv2.dilate(noisy_img,kernel,iterations = dilate_iteration)
    return img, noisy_img

def write_images(img, noisy_img, debug_img):
    global img_count

    img       = 255 - cv2.resize(img,       (0,0), fx = 1/scale_w, fy = 1/scale_h)
    noisy_img = 255 - cv2.resize(noisy_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    debug_img = 255 - cv2.resize(debug_img, (0,0), fx = 1/scale_w, fy = 1/scale_h)
    
    if img_count <= train_num:            
        cv2.imwrite(os.path.join(data_dir, train_dir, imgs_dir, '{}.jpg'.format(str(img_count).zfill(6))), img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, noisy_dir, '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, train_dir, debug_dir, '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 
    else:
        cv2.imwrite(os.path.join(data_dir, val_dir, imgs_dir, '{}.jpg'.format(str(img_count).zfill(6))), img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, noisy_dir, '{}.jpg'.format(str(img_count).zfill(6))), noisy_img) 
        cv2.imwrite(os.path.join(data_dir, val_dir, debug_dir, '{}.jpg'.format(str(img_count).zfill(6))), debug_img) 

    img_count += 1

print('\nsynthesizing image data...')
for i in tqdm(range(num_imgs)):
    # make a blank image
    img = np.ones((h, w), dtype = np.uint8)*255

    # set random parameters
    font = font_list[np.random.randint(len(font_list))]
    bottomLeftCornerOfText = (np.random.randint(word_start_x, int(img.shape[1]/3)), np.random.randint(0, int(img.shape[0]*0.8))) # (x, y)
    fontColor              = np.random.randint(0,30)
    fontScale              = np.random.randint(1800, 2400)/1000
    lineType               = np.random.randint(1,3)
    thickness              = np.random.randint(1,7)
    
    # put text
    img, y_line_list, text_height = print_lines(img, font, bottomLeftCornerOfText, fontColor, fontScale, lineType, thickness)

    # add noise
    noisy_img = get_noisy_img(img, y_line_list, text_height)

    # degrade_quality
    img, noisy_img = degrade_qualities(img, noisy_img)

    # morphological operations
    img, noisy_img =  erode_dilate(img, noisy_img)

    # make debug image
    debug_img = get_debug_image(img, noisy_img)

    # write images
    write_images(img, noisy_img, debug_img)

    '''
    cv2.imshow('textonimage', img)
    cv2.imshow('noisy_img', noisy_img)
    cv2.waitKey()
    '''
