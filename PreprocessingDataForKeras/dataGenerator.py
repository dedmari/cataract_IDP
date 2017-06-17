import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
import scipy

# original frame size
imageDim1 = 1080 
imageDim2 = 1920
parent_directory_training_labels = "../cataractsProject/train-labels-without-heading/"
parent_dir_frames = "../cataractsProject/trainFrames/"  ## directory where frames directories are stored
train_frames_folders = []
validation_frames_folders = []
test_frames_folders = []

def frames_allocation_training(number_videos_training) :
  ## inserting training label file names in a list
  for i in range(1,(number_videos_training+1)):
    frames_folder_index = str(i).zfill(2)
    train_frames_folders.insert(i, frames_folder_index)

def frames_allocation_validation(number_videos_training, number_videos_validation) :
  ## inserting validation label file names in a list
  for i in range((number_videos_training+1),((number_videos_training+1)+number_videos_validation)):
    frames_folder_index = str(i).zfill(2)
    validation_frames_folders.insert(i, frames_folder_index) # inserting folder names where frames for particular video are stored

def frames_allocation_test(number_videos_training, number_videos_validation, number_videos_test) :
  ## inserting test label file names in a list
  for i in range(((number_videos_training+1)+number_videos_validation),(((number_videos_training+1)+number_videos_validation)+number_videos_test)):
    frames_folder_index = str(i).zfill(2)
    test_frames_folders.insert(i, frames_folder_index) # inserting folder names where frames for particular video are stored

def training_generator(batch_size, number_videos_training): 
  frames_allocation_training(number_videos_training)

  ## intializing input array for storing video frames for a batch
  X    = np.zeros(shape=(batch_size, imageDim1 / 2,imageDim2 / 2, 3))# resizing dimensions by 0.5 and 3 is color channel dimension
  Y    = np.zeros(shape=(batch_size, 21)) # 21 is number of classes
  batch = 0
  batch_index = 0
  while True:
    ## traversing all the folders containing training video frames
    for frames_folder in train_frames_folders :  
      root_dir = parent_dir_frames+'trainingFrames'+frames_folder+'/'
      train_labels = np.loadtxt(parent_directory_training_labels+'train'+frames_folder+'.csv', delimiter=",") 
      label_index = 0 
      print("training frames folders name : " + root_dir)
      print("training label file name ",parent_directory_training_labels+'train'+frames_folder+'.csv')
      all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #all image in a folder paths are saved in list
      for img_path in all_img_paths:
        img      	      = load_img(img_path)
        imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
        resized_imgArray      = scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
        X[batch_index] 	      = resized_imgArray
        Y[batch_index]        = train_labels[label_index,:]
        batch_index += 1
        label_index += 1
        if batch_index == batch_size :
          batch_index = 0
          batch +=1
          print("now yielding batch", batch)
          yield X, Y

def validation_generator(batch_size, number_videos_training, number_videos_validation): 
  frames_allocation_validation(number_videos_training, number_videos_validation)

  ## intializing input array for storing video frames for a batch
  X    = np.zeros(shape=(batch_size, imageDim1 / 2,imageDim2 / 2, 3))# resizing dimensions by 0.5 and 3 is color channel dimension
  Y    = np.zeros(shape=(batch_size, 21)) # 21 is number of classes	
  batch = 0
  batch_index = 0
  while True:
    ## traversing all the folders containing training video frames
    for frames_folder in validation_frames_folders :  
      root_dir = parent_dir_frames+'trainingFrames'+frames_folder+'/'
      validation_labels = np.loadtxt(parent_directory_training_labels+'train'+frames_folder+'.csv', delimiter=",") 
      label_index = 0 
      print("validation frames folders name : " + root_dir)
      print("validation label file name ",parent_directory_training_labels+'train'+frames_folder+'.csv')
      all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #all image in a folder paths are saved in list
      for img_path in all_img_paths:
        img      	      = load_img(img_path)
        imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
        resized_imgArray      = scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
        X[batch_index] 	      = resized_imgArray
        Y[batch_index]        = validation_labels[label_index,:]
        batch_index += 1
        label_index += 1
        if batch_index == batch_size :
          batch_index = 0
          batch +=1
          print("now yielding batch", batch)
          yield X, Y

def test_generator(batch_size, number_videos_training, number_videos_validation, number_videos_test): 
  frames_allocation_test(number_videos_training, number_videos_validation, number_videos_test)
  ## intializing input array for storing video frames for a batch
  X    = np.zeros(shape=(batch_size, imageDim1 / 2,imageDim2 / 2, 3))# resizing dimensions by 0.5 and 3 is color channel dimension
  Y    = np.zeros(shape=(batch_size, 21)) # 21 is number of classes	
  batch = 0
  batch_index = 0
  while True:
    ## traversing all the folders containing training video frames
    for frames_folder in test_frames_folders :  
      root_dir = parent_dir_frames+'trainingFrames'+frames_folder+'/'
      test_labels = np.loadtxt(parent_directory_training_labels+'train'+frames_folder+'.csv', delimiter=",") 
      label_index = 0 
      print("test frames folders name : " + root_dir)
      print("test label file name ",parent_directory_training_labels+'train'+frames_folder+'.csv')
      all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpeg'))) #all image in a folder paths are saved in list
      for img_path in all_img_paths:
        img      	      = load_img(img_path)
        imgArray 	      = img_to_array(img) #converting image to numpy array. Shape will be as that of image
        resized_imgArray      = scipy.ndimage.zoom(imgArray, (0.5,0.5,1)) #resizing image
        X[batch_index] 	      = resized_imgArray
        Y[batch_index]        = test_labels[label_index,:]
        batch_index += 1
        label_index += 1
        if batch_index == batch_size :
          batch_index = 0
          batch +=1
          print("now yielding batch", batch)
          yield X, Y
