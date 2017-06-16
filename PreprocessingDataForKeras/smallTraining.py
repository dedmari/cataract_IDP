#from keras.models import Sequential
from keras.models import *
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import *
#from loadCataractData import * # used for testing on small data set
from finalLoadCataractData import *
from keras.optimizers import SGD

epochs = 100
X, y, img_width, img_height = load_X_y()
#y  = y[0:275, :] ## for testing on 275 frames. Remove it after testing on small data set
print "Training.."

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dense(512)) # It's size matters a lot during training
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(21))
model.add(Activation("sigmoid")) ## sigmoid is good for binary classification and when training label dont have effect on other. We can use other activation also e.g softmax

# let's train the model using SGD + momentum.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', #Also we can use categorical_crossentropy
              optimizer='sgd',       # rmsprop can also be used
             metrics=['accuracy'])
model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
model.save_weights('first_try.h5')
preds = model.predict_proba(X[7:9,:])
