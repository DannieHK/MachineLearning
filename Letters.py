import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from matplotlib import pyplot

!rm -r machine-learning
!git clone https://github.com/joneikholmkea/machine-learning

train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    shear_range = 0.2,
    zoom_range = 0.2,
    rotation_range = 20,
    width_shift_range =0.2,
    height_shift_range = 0.2
)

training_set = train_datagen.flow_from_directory (
    '/content/machine-learning/img/letters/AB/train',
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale' # 'rgb' is for color, if needed
)
print(training_set.class_indices)

test_set = train_datagen.flow_from_directory (
    '/content/machine-learning/img/letters/AB/test',
    target_size = (28,28),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale' # 'rgb' is for color, if needed.
)
print(training_set.class_indices)

model = Sequential()
model.add(Conv2D(filters = 5,
                 kernel_size = 3,
                 activation = 'relu',
                 input_shape = [28,28,1]) # input_shape is the size of the image (pixels), 1 if it's grayscale, 3 if it's rgb.
)

model.add(MaxPool2D(pool_size = 2, strides = 2))
model.add(Flatten())
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(units = 10, activation='relu'))
model.add(Dense(units = 2, activation='softmax'))
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(training_set, epochs = 1)
model.evaluate(test_set)
