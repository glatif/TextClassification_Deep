
# Further data pre-processing before Machine Learning

# Resize images to 64X64
# Convert images to grayscale

import os
from os import path
import cv2

# for every user
for userid in range(1,53):
    username="user" + format(userid,'03d')
    print(username)
    #input_dir = path.join("dataset","dataset",username)
    training_dir = path.join("Senior Project","training",username)
    validation_dir = path.join("Senior Project","validation",username)
    test_dir = path.join("Senior Project","test",username)
    assert path.exists(training_dir)
    assert path.exists(validation_dir)
    assert path.exists(test_dir)

    
    # for every file in the training dir, resize and convert to gray scale
    input_dir = training_dir
    for file in os.listdir(input_dir):
        original_image = cv2.imread(os.path.join(input_dir,file))
        #cv2.imshow('original',original_image)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_image, (64,64))
        #cv2.imshow('resized', resized_img)
        cv2.imwrite(os.path.join(input_dir,file), resized_img)

    # for every file in the validation dir, resize and convert to gray scale
    input_dir = validation_dir
    for file in os.listdir(input_dir):
        original_image = cv2.imread(os.path.join(input_dir,file))
        #cv2.imshow('original',original_image)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_image, (64,64))
        #cv2.imshow('resized', resized_img)
        cv2.imwrite(os.path.join(input_dir,file), resized_img)
                
    # for every file in the test dir, resize and convert to gray scale
    input_dir = test_dir
    for file in os.listdir(input_dir):
        original_image = cv2.imread(os.path.join(input_dir,file))
        #cv2.imshow('original',original_image)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_image, (64,64))
        #cv2.imshow('resized', resized_img)
        cv2.imwrite(os.path.join(input_dir,file), resized_img)
        