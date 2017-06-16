import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
from skimage import transform as tf
import matplotlib.pyplot as plt
import scipy

def load_X_y():
	train_labels = np.loadtxt("../cataractsProject/train-labels/train01WithoutHeading.csv", delimiter=",")
	root_dir = '../extractFramesScript/frames01/'
	imgs = []
	labels = []
	
	all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #image paths saved in list
	totalImages =  len(all_img_paths)
	img = load_img(all_img_paths[0])
	imgArray = img_to_array(img) #converting image to numpy array. Shape will be as that of image dimensions i.e e.g 1080, 1920, 3
	imageDims = imgArray.shape; # return tuple conatining dimension of an image
	del all_img_paths[0] # deleting first image path so that it will not be used in loop again
	small_train_data = 10 # machine is not able to store all the available images. it shows memory error. Thats why small size of 	training images
	X    = np.empty(shape=(small_train_data,imageDims[0] / 2,imageDims[1] / 2,imageDims[2])) 
	resized_imgArray  =   scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
	X[0] 		  = resized_imgArray
	imageFromArray    = array_to_img(resized_imgArray)
	print len(all_img_paths)
	print "------------------"
	
	count = 1
	for img_path in all_img_paths:
	    print count
	    if count > small_train_data - 1: #for small set of training data
		break
	    img      	      = load_img(img_path)
	    imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
	    resized_imgArray  =   scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
	    X[count] = resized_imgArray
	    count +=1
	y = train_labels[0:small_train_data,:]# selecting only small lebels for small training data
	return X, y, imageDims[0] / 2, imageDims[1] / 2 # Image dimensions are divided by 2 because it is resized to half of original image size
