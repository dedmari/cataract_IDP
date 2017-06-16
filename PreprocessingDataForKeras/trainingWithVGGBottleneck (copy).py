######
## In this model, I am freezing all the layers of VGG16 till fully connected layer, So that it runs compute efficiently on CPU
####
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation
from keras import applications
from loadCataractData import *
from keras.optimizers import SGD
top_model_weights_path = 'fc_model.h5'
epochs = 8
batch_size = 32
nb_train_samples = 8
nb_validation_samples = 2
print "Loading Training Data.."
X, y, img_width, img_height = load_X_y() #loading training data

def save_bottlebeck_features():
	# build the VGG16 network
    	model = applications.VGG16(include_top=False, weights='imagenet')
	#model.summary()
	bottleneck_features_train = model.predict(X[0:nb_train_samples,:]) # checking prediction of vgg16 on training data (only first 8 samples taken into consideration)
	np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
	bottleneck_features_validation = model.predict(X[nb_train_samples:10,:]) # validation data giving to VGG model
    	np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

def train_top_model():
    print "Training.."
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = y[0:nb_train_samples,:]
    print "----------------------------------------------------------"
    print "----------------------------------------------------------"
    print "training labels data shape"
    print train_labels.shape
    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = y[nb_train_samples:10,:]
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=train_data.shape[1:])) # It's not good approach to use convolution layer after getting features from pre-trained model. Replace it by pooling layer.
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    #model.add(Flatten(input_shape=train_data.shape[1:]))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(GlobalAveragePooling2D()(modelOutput))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(21, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    #model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    model.save_weights(top_model_weights_path)


#save_bottlebeck_features()
train_top_model()
print "working"
