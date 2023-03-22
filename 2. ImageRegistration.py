#Adapted from https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
import numpy as np
import imutils
import cv2

def align_image(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # convert both the input image and template to grayscale
    imageGray = image.copy()
    templateGray = template.copy()
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                                     matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    # return the aligned image
    return aligned

basepath = 'G:/Mi unidad/2023/Doctorado/60367-Tutela/Paper Dataset/SAR_Toronto/' #Path of folder where the image is located
imgpath = 's1a-iw-grd-vv-20220929t230848-20220929t230913-045226-0567e1-001_scaled.tiff' #Scaled image name and extension

#Start
imgtemplatepath = 's1a-iw-grd-vv-20220824t230846-20220824t230911-044701-055641-001_scaled.tiff' #Scaled image of reference
template = cv2.imread(basepath + imgtemplatepath, cv2.IMREAD_GRAYSCALE)  # Reference image.

print("[INFO] loading images...")
image = cv2.imread(basepath + imgpath, cv2.IMREAD_GRAYSCALE)  # Image to be aligned.
image = cv2.resize(image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)

# align the images
print("[INFO] aligning images...")
aligned = align_image(image, template, maxFeatures=500, debug=False)
aligned = cv2.resize(aligned, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)

#Shape verification
print('Shape before alignment: ', image.shape)
print('Shape after alignment: ', aligned.shape)

#Save image to disk
pathsplit = imgpath.split('.tiff')
imgpathregist = basepath + pathsplit[0] + '_registered.tiff'
cv2.imwrite(imgpathregist, aligned)

def calculate_mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    return mse

print('MSE Before Registration: ', calculate_mse(template, image))
print('MSE After Registration: ', calculate_mse(template, aligned))