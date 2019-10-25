import cv2
import numpy as np
 
img = cv2.imread('frames/frame1.jpg')
height, width, channels = img.shape
fr = 24
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fr, (width,height))
count = 0
nframes = 300
while(count<nframes):
  frame = cv2.imread('frames/frame'+str(count)+'.jpg')
  cv2.imshow( "frames", frame );
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
  out.write(frame)
  count+=1
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 