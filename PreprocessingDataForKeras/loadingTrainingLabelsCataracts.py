import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
from skimage import transform as tf
import matplotlib.pyplot as plt
import scipy
train_labels = np.loadtxt("../cataractsProject/train-labels/train01WithoutHeading.csv", delimiter=",")

root_dir = '../extractFramesScript/frames01/'
imgs = []
labels = []
###########
#img = load_img('../frames01/train01_3252.jpeg')  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

###########
#all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpeg')) # getting images from all video frames
all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #image paths saved in list
totalImages =  len(all_img_paths)
#print totalImages
img = load_img(all_img_paths[0])
imgArray = img_to_array(img) #converting image to numpy array. Shape will be as that of image dimensions i.e e.g 1080, 1920, 3
imageDims = imgArray.shape; # return tuple conatining dimension of an image
del all_img_paths[0] # deleting first image path so that it will not be used in loop again
#X    = np.empty(shape=(totalImages,imageDims[0],imageDims[1],imageDims[2])) # used to store data of all images. One image data in one row
small_train_data = 500 # machine is not able to store all the available images. it shows memory error. Thats why small size of training images
X    = np.empty(shape=(small_train_data,imageDims[0] / 2,imageDims[1] / 2,imageDims[2])) 
#X[0] = imgArray # storing array corresponding to first image
#new_imgArray = tf.resize(imgArray , (600, 300, 3), order=0)
resized_imgArray  =   scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
X[0] 		  = resized_imgArray
imageFromArray    = array_to_img(resized_imgArray)
plt.figure("original")
plt.imshow(img)
plt.figure("resized")
plt.imshow(imageFromArray, interpolation='nearest')
plt.show()
#print resized_imgArray.shape # image resized by 0.5 factor along pixels and color channel remains same
#print imgArray
print len(all_img_paths)
print "------------------"

#print resized_imgArray
#print len(all_img_paths)
count = 1
for img_path in all_img_paths:
    print count
    img      	      = load_img(img_path)
    imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
    resized_imgArray  =   scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
    X[count] = resized_imgArray
    #print imgArray.shape
    #imgs.append(imgArray)
    count +=1
    if count > 459: #for small set of training data
	break
#    break
    #imgs.append(img)prin
    #label = get_class(img_path)
    #labels.append(label)

print train_labels.shape
print X.shape

print X[458]
print "-------------------------------END---------------------------------------------"
#print imgs.shape
#print dataset.shape
#print "-------------"

