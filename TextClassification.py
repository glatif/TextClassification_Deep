import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import keras
import keras.utils
from keras import utils as np_utils
#Start
train_data_path = '/etc/train/'
validation_data_path = '/etc/validate/'
img_rows = 64
img_cols = 64
epochs = 80
batch_size = 32
#num_of_train_samples = 10400
#num_of_test_samples = 2600
#Image Generator
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                     target_size=(img_rows, img_cols),
                                                     color_mode = 'grayscale',
                                                     batch_size=batch_size,
                                                     class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_path,
                                                         target_size=(img_rows, img_cols),
                                                         color_mode = 'grayscale',
                                                         batch_size=batch_size,
                                                         class_mode='categorical')
# Build model
model = Sequential()
model.add(Convolution2D(64, (3, 3), input_shape=(img_rows, img_cols, 1),  padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Convolution2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(52))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
 

#Train
history = model.fit_generator(train_generator,
                    steps_per_epoch=(train_generator.samples // batch_size) + 1,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=(validation_generator.samples // batch_size) + 1)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# ===========================================================================
# Once you are satisfied with validation results (accuracy) perform testing using test dataset
# ===========================================================================
#Testing
test_data_path = '/etc/test/'
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_data_path,
                                                         target_size=(img_rows, img_cols),
                                                         color_mode = 'grayscale',
                                                         batch_size=batch_size,
                                                         class_mode='binary')

predictions = model.predict_generator(test_generator,  (test_generator.samples // batch_size) + 1)
predicted_classes = np.argmax(predictions, axis=1)
# Generate Confution Matrix and Classification Report for test data
true_classes = test_generator.classes[test_generator.index_array]
test_accuracy = accuracy_score(true_classes, predicted_classes)
test_accuracy
test_cm = confusion_matrix(true_classes, predicted_classes)
class_labels = list(test_generator.class_indices.keys())   
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

# Finally save your model in h5 file. This file can be read later on to retrieve 
# saved model (as you will not retrain the model for use in production system)

# Saving model
# Save your optimal model
model.save("cnn_model.h5")

# Loading Model (This may be done on your device, say raspberry pi etc)
# Load your model later and test on user input

model = keras.models.load_model("cnn_model.h5")

# Write code here to test your production input