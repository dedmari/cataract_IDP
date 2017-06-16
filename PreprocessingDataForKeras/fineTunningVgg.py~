from keras import applications
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation
from keras.optimizers import SGD
from keras import backend as K
from loadCataractData import *
# path to the model weights files.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

 #loading training data
X, y, img_width, img_height = load_X_y()
print "Data loaded succesfully!"
epochs = 8
batch_size = 16
nb_train_samples = 8
nb_validation_samples = 2

train_data = X[0:nb_train_samples,:]
train_labels = y[0:nb_train_samples,:]
validation_data = X[nb_train_samples:10,:]
validation_labels = y[nb_train_samples:10,:]
# build the VGG16 network
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
#top_model = Sequential()
#top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))

#################
top_model = Sequential()
top_model.add(Conv2D(64, (3, 3), input_shape=base_model.output_shape[1:]))
top_model.add(Activation('relu'))
top_model.add(GlobalAveragePooling2D())
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(21, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#################

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

#base_model.add(top_model)
base_model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in base_model.layers[:25]:
    layer.trainable = False


base_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# fine-tune the model
base_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
