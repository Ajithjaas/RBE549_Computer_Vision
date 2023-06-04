import cv2
import numpy as np
import glob

a = sorted(glob.glob(r'Data/Images/*.png'))
b = sorted(glob.glob(r'Data/SegImages/*.png'))
img_array = []

for filename1,filename2 in zip(a,b):
    # RGB Images
    img1 = cv2.imread(filename1)
    img1 = np.array(img1)

    # Segmented Images
    img2 = cv2.imread(filename2)
    img2 = np.array(img2)

    # Combining images
    img = np.vstack((img1,img2))
    height, width, channel = img.shape
    img_array.append(img)

out = cv2.VideoWriter('Data/Results/Semantic_Segmentation.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 3, (width, height))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()