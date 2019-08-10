from port.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="this is input")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
org = image.copy()
image = imutils.resize (image ,height = 500)

gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

print ("Step1 : edge detction")
cv2.imshow("Image ", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


for c in cnts :
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx 
        break

print("Step 2 find the contours ")
cv2.drawContours(image, [screenCnt], 0 , (0, 255, 0), 3)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(org, screenCnt.reshape(4, 2) * ratio)


warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

print("this is the steo 3")
cv2.imshow("Original", imutils.resize(org, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.imwrite("output3.png", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
