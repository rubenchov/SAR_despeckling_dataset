import cv2
import numpy as np
basepath = 'G:/Mi unidad/2023/Doctorado/60367-Tutela/Paper Dataset/SAR_Toronto/' #Path of folder where the image is located

#List of images
img1path = 's1a-iw-grd-vv-20220824t230846-20220824t230911-044701-055641-001_scaled.tiff' #Image not aligined (registered) because it was used as the reference to register the rest
img2path = 's1a-iw-grd-vv-20220905t230847-20220905t230912-044876-055c29-001_scaled_registered.tiff'
img3path = 's1a-iw-grd-vv-20220917t230847-20220917t230912-045051-056208-001_scaled_registered.tiff'
img4path = 's1a-iw-grd-vv-20220929t230848-20220929t230913-045226-0567e1-001_scaled_registered.tiff'
img5path = 's1a-iw-grd-vv-20221011t230847-20221011t230912-045401-056dca-001_scaled_registered.tiff'
img6path = 's1a-iw-grd-vv-20221023t230848-20221023t230913-045576-0572f6-001_scaled_registered.tiff'
img7path = 's1a-iw-grd-vv-20221104t230847-20221104t230912-045751-0578dc-001_scaled_registered.tiff'
img8path = 's1a-iw-grd-vv-20221116t230847-20221116t230912-045926-057ecb-001_scaled_registered.tiff'
img9path = 's1a-iw-grd-vv-20221128t230847-20221128t230912-046101-0584b4-001_scaled_registered.tiff'
img10path = 's1a-iw-grd-vv-20221222t230845-20221222t230910-046451-0590a5-001_scaled_registered.tiff'

pathlist = [basepath + img1path,
               basepath + img2path,
               basepath + img3path,
               basepath + img4path,
               basepath + img5path,
               basepath + img6path,
               basepath + img7path,
               basepath + img8path,
               basepath + img9path,
               basepath + img10path]

#print(rutaimglist[0])
img = cv2.imread(pathlist[0])  #Image load
height, width, depth = img.shape
imgacum = np.zeros((height, width), np.single)
imgscalar = 10.0 * np.ones((height, width), np.single)

for imgpath in pathlist:
    print(imgpath)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)  # Image load
    img = img.astype(np.single)
    imgacum = cv2.add(imgacum, img)
avgGT = cv2.divide(imgacum, imgscalar)
avgGT = avgGT.astype(np.uint8)
cv2.imwrite(basepath + 'AverageGT.tiff', avgGT)