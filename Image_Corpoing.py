#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from os import path
from PIL import Image
from PIL import ImageOps
from os import listdir
from os.path import isfile, join

index = 1;
input_dir_name = "input_047"
cropped_dir_name = os.path.join(input_dir_name,"cropped")
onlyfiles = [f for f in listdir(input_dir_name) if isfile(join(input_dir_name, f)) and '.jpg' in f]   

for file in onlyfiles:
    # Read the fixed form as image
    image=cv2.imread(os.path.join(input_dir_name, file))
    if (not path.exists(cropped_dir_name)): os.mkdir(cropped_dir_name)
#sig1
    cropped_image = image[1060:1200, 120:575]
    filename = "%d_sig.jpg"%(index)
    cv2.imwrite(os.path.join(cropped_dir_name, filename), cropped_image)
    index+=1;
#sig2
    cropped_image = image[1060:1200, 657:1115]
    filename = "%d_sig.jpg"%(index)
    cv2.imwrite(os.path.join(cropped_dir_name, filename), cropped_image)
    index+=1;
#client_sig
    cropped_image = image[1325:1470, 130:600]
    filename = "%d_sig.jpg"%(index)
    cv2.imwrite(os.path.join(cropped_dir_name, filename), cropped_image)
    index+=1;
    
# WHITE SPACE REMOVAL
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

for file_name in os.listdir(cropped_dir_name):
    print (file_name)
    image=Image.open(os.path.join(cropped_dir_name, file_name))
    image.load()
    imageSize = image.size
    # Crop center piece of the image containg the letter
    image = crop_center(image, 450, 140)
    # Remove any whitespace surrounding the image
    
    # remove alpha channel
    invert_im = image.convert("RGB") 
    
    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()
    
    cropped=image.crop(imageBox)
    #print ("%s Size:%s New Size:%s"%(file_name, imageSize, imageBox))
    cropped.save(os.path.join(cropped_dir_name, file_name))
               
#IMAGE PREPROCESSING
for file in os.listdir(cropped_dir_name):
        original_image = cv2.imread(os.path.join(cropped_dir_name,file))
        #cv2.imshow('original',original_image)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_image, (64,64))
        #cv2.imshow('resized', resized_img)
        cv2.imwrite(os.path.join(cropped_dir_name,file), resized_img)
