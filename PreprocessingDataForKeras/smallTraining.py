#from keras.models import Sequential
from keras.models import *
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import *
from dataGenerator import *
#from dataGeneratorOld import *
from keras.optimizers import SGD
import numpy as np

#Data distribution for training, validation and test
number_videos_training = 19 # 1-19 Videos
number_videos_validation = 5 # 20-24 Videos
number_videos_test = 1 # 25 Video

epochs = 100
batch_size = 32
steps_per_epoch_tr = 371209 / batch_size # It should typically be equal to the number of unique samples in dataset(1-19) divided by the batch size.
validation_steps = 82839 / batch_size #15 / batch_size # total number of frames (20-24)/ batch_size
test_steps = 40820 / batch_size # total number of frames in test videos(25)
img_width = 1080 / 2 # scalling by 0.5
img_height = 1920 / 2
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
model.fit_generator(
    training_generator(batch_size, number_videos_training),
    validation_data=validation_generator(batch_size, number_videos_training, number_videos_validation),
    steps_per_epoch=steps_per_epoch_tr,
    validation_steps=validation_steps,
    epochs=epochs,
    #callbacks=[PrintBatch()],
    verbose=1)
model.save_weights('small_model_weights.h5')
pred = model.predict_generator(test_generator(batch_size, number_videos_training, number_videos_validation, number_videos_test), steps = test_steps, verbose=1)

np.savetxt(
    'predictions/predictions_25.csv',          # file name
    pred,  # array to save
    fmt='%.9f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    header= 'probability')   # file header
#preds = model.predict_proba(X[7:9,:]) # instead of X[7:9,:] have to give text data for prdiction and gives output as probability
