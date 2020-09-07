#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import keras.utils
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

index = 1;
input_dir_name = "input_047"
cropped_dir_name = os.path.join(input_dir_name,"cropped")


#RELOAD THE MODEL
model = keras.models.load_model("/etc/cnn_model.h5")

#TESTING
img_rows = 64
img_cols = 64
batch_size = 1

#test_data_path = os. getcwd()
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(input_dir_name,
                                                         target_size=(img_rows, img_cols),
                                                         color_mode = 'grayscale',
                                                         batch_size=batch_size,
                                                         class_mode=None,
                                                         shuffle = False)
predictions = model.predict_generator(test_generator,  (test_generator.samples // batch_size) )
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = predicted_classes + 1
print(predicted_classes)

print('Top Left User: {0}\n'.format(predicted_classes[0]))
print('Top Right User: {0}\n'.format(predicted_classes[1]))
print('Bottom Left User: {0}\n'.format(predicted_classes[2]))

file = open("Classes.txt", "w") 
file.write("Predicted classes are:\n")
file.write('Top Left User: {0}\n'.format(predicted_classes[0]))
file.write('Top Right User: {0}\n'.format(predicted_classes[1]))
file.write('Bottom Left User: {0}\n'.format(predicted_classes[2]))
#file.write(predicted_classes) 
file.close() 
