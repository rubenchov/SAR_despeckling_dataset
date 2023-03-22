import cv2
import numpy as np

basepath = 'G:/Mi unidad/2023/Doctorado/60367-Tutela/Prueba SAR Original/Img_Toronto/' #Path of folder where the image is located
imgpath = 's1a-iw-grd-vv-20220824t230846-20220824t230911-044701-055641-001.tiff' #Image name and extension

img = cv2.imread(basepath + imgpath, cv2.IMREAD_UNCHANGED) #Load image

#Comment the following 5 lines if you don't need them
print('Image shape: ', img.shape)
print('Image type: ', img.dtype)
print('Image max pixel value: ', np.max(img))
print('Image mean pixel value: ', np.mean(img))
print('Image min pixel value: ', np.min(img))

img2 = img.astype(np.single) #Change datatype to real values
escala_display = np.mean(img2) * 3.0 #Mean value times 3
min = np.min(img2) #Calculation of minimum value of pixel of all the image
img2[img2 > escala_display] = escala_display #Values higher or equal to mean*3 are reasigned to mean*3
img2[img2 < min] = 0 #Values lower than min(img) are reasigned to zero. Other values will remain the same
img3 = 255.0 * (img2 / escala_display) #Normalized to 0-1 and then rescaled 0-255
img4 = img3.astype(np.uint8) #Change datatype to 8-bit unsigned integer

pathsplit = imgpath.split('.tiff') #Split the path of the image [name, '.tiff']
imgpathscaled = basepath + pathsplit[0] + '_scaled.tiff' #Added string '_scaled.tiff' to new name of image to save
cv2.imwrite(imgpathscaled, img4)
print('Shape of rescaled image: ', img4.shape)
