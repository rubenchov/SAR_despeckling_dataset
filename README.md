# SAR despeckling dataset
SAR dataset for despeckling, including actual SAR images and ground truth.

Procedure:
1. You must have a SAR image in .tiff format downloaded from Sentinel-1. This script is made to rescale intensity in L1 Detected High-Res Dual-Pol (GRD-HD) images downloaded from https://search.asf.alaska.edu/#/.
2. Run the script 1. Rescale Intensity.py to rescale the intensity of an image. A new image with the name "_scaled.tiff" will be generated. It has to be done in every image.
3. Image registration is recommended by using ORB as shown in https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/ (Use 2. ImageRegistration.py)
4. The ground truth image will be the same size as the reference image, by averaging the images obtained in the previous step (Use 3. Generate_GTruth.py). 
5. Image clip is performed by firs defining three parameters: height, width and step. The default is height=width=step=512. The images (noisy and ground truth) will be loaded and then cropped in two folders named 'Gtruth' and 'Noisy'. Depending of the setting of the parameters, the resulting dataset will be bigger or smaller. These two folders can be used for training deep learning models. (Use 4. Clip_Image.py)
