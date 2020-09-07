
# Traing/Test split for user identification based on words

# For every user create a subfolder in training and testing folders
# Go through each user subfolder in dataset folder
# Move 80% images to training and 20% to test

import os
from os import path
from natsort import natsorted
import shutil

# for every user
for userid in range(1,21):
    username="user" + format(userid,'03d')
    print(username)
    input_dir = path.join("Senior Project","dataset",username)
    output_training_dir = path.join("Senior Project","training",username)
    output_validation_dir = path.join("Senior Project","validation",username)
    output_test_dir = path.join("Senior Project","test",username)
    if (not path.exists(output_training_dir)): os.mkdir(output_training_dir)
    if (not path.exists(output_validation_dir)): os.mkdir(output_validation_dir)
    if (not path.exists(output_test_dir)): os.mkdir(output_test_dir)

    all_files = os.listdir(input_dir)
    # file names are placed under natural ordering
    all_files = natsorted(all_files)
    counter = 1
    # for every 10 files in the user dir, copy the first 8 files in training and 
    # the next 2 files in test folder.
    for file in all_files:
        if (counter > 6 and counter <= 8):
            shutil.copy(path.join(input_dir,file), path.join(output_validation_dir,file))
        elif (counter > 8):
            shutil.copy(path.join(input_dir,file), path.join(output_test_dir,file))
        else:
            shutil.copy(path.join(input_dir,file), path.join(output_training_dir,file))
        counter = counter + 1
        if (counter == 11): counter = 1
