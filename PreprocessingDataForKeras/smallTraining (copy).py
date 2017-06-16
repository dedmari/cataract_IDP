#from keras.models import Sequential
from keras.models import *
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import *
from loadCataractData import *
from keras.optimizers import SGD


X, y, img_width, img_height = load_X_y() #loading training data
print "Training.."
# dimensions of images.
#img_width, img_height = 540, 960

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
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalMaxPooling2D())
## Fully connected layer
#model.add(Flatten())
#model.add(Dense(64))
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
# saving checkpoint with training weights
#checkpoint_path="train01weights.{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
#checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
#model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])
model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2, verbose=1)
model.save_weights('first_try.h5')
#preds = model.predict(X[7:9,:])
preds = model.predict_proba(X[7:9,:])
#print "prediction values are:"
#preds[preds>0.5] = 1
#preds[preds==0.5] = 0.5
#preds[preds<0.5] = 0
#print preds
