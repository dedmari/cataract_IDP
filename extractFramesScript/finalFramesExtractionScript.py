import os
import pylab
import imageio
from PIL import Image
#import Image
#filename = './train01.mp4'
parentDirectoryVideos        = "../cataractsProject/train/"
parentDirectoryStoringFrames = "../cataractsProject/trainFrames/"
trVideoNames  = []
## inserting video names in a list
for i in range(1,26):
  videoName = 'train'+ str(i).zfill(2) + '.mp4' # zfill is used to include trailing zeros in case integer is less than 2 digts
  trVideoNames.insert(i, videoName)

print "Extracting frames from videos:"
for videoName in trVideoNames :
  print videoName 
  filename  = parentDirectoryVideos+videoName
  #directoryStoringFrames =  './trainingFrames'+videoName[5:7]+'/'
  directoryStoringFrames = parentDirectoryStoringFrames + 'trainingFrames'+videoName[5:7]+'/'
  print "Frames directory path: "+ directoryStoringFrames
  vid = imageio.get_reader(filename,  'ffmpeg')
  for i, image in enumerate(vid): # iterating through all the frames in a video
    numstr = str(i)
    #print numstr
    im = Image.fromarray(image)  
    if not os.path.exists(directoryStoringFrames):
    	os.makedirs(directoryStoringFrames)
    im.save(directoryStoringFrames+videoName+"_"+numstr+".jpeg")
   # if i == 10 :
    #	break 
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(num), fontsize=20)
    #pylab.imshow(image)
#pylab.show()
print "All frames extracted successfully"
