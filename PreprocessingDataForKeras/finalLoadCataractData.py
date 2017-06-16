import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
from skimage import transform as tf
import matplotlib.pyplot as plt
import scipy
def load_X_y():
        parent_directory_training_labels = "../cataractsProject/train-labels-without-heading/"
        train_labels_filenames = []
        train_frames_folders = [] 
        train_frames_folders.insert(1,'trainingFrames01')## folder name in which frames of first training video are stored

## inserting training label file names in a list
	for i in range(2,26):
  	  labelFileName = 'train'+ str(i).zfill(2) + '.csv' # zfill is used to include trailing zeros in case integer is less than 2 digts
  	  train_labels_filenames.insert(i, labelFileName)
	  frames_folder = 'trainingFrames'+ str(i).zfill(2)
          train_frames_folders.insert(i, frames_folder)
## loading labels for first video
	train_labels = np.loadtxt(parent_directory_training_labels+"train01.csv", delimiter=",")

## Appending rest of data labels (except for first video) into one file
        for labelFileNames in train_labels_filenames :
	  print "traning data label file name: "+labelFileNames
	  currentFile = np.loadtxt(parent_directory_training_labels+labelFileNames, delimiter=",")
          train_labels = np.append(train_labels,currentFile, axis = 0)

	totalFrames = train_labels.shape[0] ## total number of frames = number of training label data 
        print "total Frames = "+str(totalFrames)


## assembling frames into one ndArray
        parent_dir_frames = "../cataractsProject/trainFramesFinal/"  ## directory where frames directories are stored
        # original frame size
        imageDim1 = 1080 
	imageDim2 = 1920

        ## intializing input array for storing all video frames
        X    = np.empty(shape=(totalFrames, imageDim1 / 2,imageDim2 / 2, 3))# resizing dimensions by 0.5 and 3 is color channel dimension
        #X    = np.empty(shape=(275, imageDim1 / 2,imageDim2 / 2, 3)) ## used only for test purpose with only 275 frames. Remove it after testing
	index = 0	
        ## traversing all the folders containing training video frames
	for frames_folder in train_frames_folders :  
          root_dir = parent_dir_frames+frames_folder+'/'
	  print "training frames folders name : " + root_dir
	  all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #all image in a folder paths are saved in list
	  for img_path in all_img_paths:
	   # if count > small_train_data - 1: #for small set of training data
	   #	break
	    img      	      = load_img(img_path)
	    imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
	    resized_imgArray  = scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
	    X[index] 	      = resized_imgArray
	    index +=1
        #print X.shape[0], " " , train_labels.shape[0]
	return X,train_labels, imageDim1 / 2, imageDim2 / 2
#X,train_labels,Dim1,Dim2 = load_X_y()
#print(X.shape)
#print(train_labels.shape) 
