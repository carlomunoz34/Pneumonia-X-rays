import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config) 

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Data preprocessing
batch_size = 32
IMG_SIZE = 150

classes = ['NORMAL', 'PNEUMONIA']
train_path = '../../Datasets/chest_xray/train'
test_path = '../../Datasets/chest_xray/test'

data_gen = ImageDataGenerator(rescale=1./255)
train_batches = data_gen.flow_from_directory(train_path, target_size=(IMG_SIZE, IMG_SIZE), classes=classes, class_mode='binary', batch_size=batch_size)
test_batches = data_gen.flow_from_directory(test_path, target_size=(IMG_SIZE, IMG_SIZE), classes=classes, class_mode='binary')

#Model
#VGG16
lr = 0.0001
beta_1 = 0.9
beta_2 = 0.999
epochs = 5
steps_per_epoch = 663
validation_steps = 624
DROP_PROB = 0.4

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=train_batches.image_shape))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Dropout(DROP_PROB))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Dropout(DROP_PROB))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Dropout(DROP_PROB))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Dropout(DROP_PROB))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=(2, 2)))
model.add(Dropout(DROP_PROB))

model.add(Flatten())

model.add(Dense(4090, activation='relu'))
model.add(Dropout(DROP_PROB))
model.add(BatchNormalization())
model.add(Dense(4090, activation='relu'))
model.add(Dropout(DROP_PROB))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

#Compile the model
optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Training
history = model.fit_generator(epochs=5, shuffle=True, generator=train_batches,
        validation_data=test_batches, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()