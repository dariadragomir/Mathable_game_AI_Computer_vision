import numpy as np
import cv2
import matplotlib.pyplot as plt #plt.plot(x,y) plt.show()
import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from numpy.random import uniform
import numpy as np

def show_imag1e(title,image):
    image=cv.resize(image,(0,0),fx=0.2,fy=0.2)
    cv.imshow(title,image)
def find_color_values_using_trackbar(frame):

	frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	#frame_hsv=frame
	def nothing(x):
		pass

	cv.namedWindow("Trackbar") 
	cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
	cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
	cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
	cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
	cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
	cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)

	while True:
		l_h = cv.getTrackbarPos("LH", "Trackbar")
		l_s = cv.getTrackbarPos("LS", "Trackbar")
		l_v = cv.getTrackbarPos("LV", "Trackbar")
		u_h = cv.getTrackbarPos("UH", "Trackbar")
		u_s = cv.getTrackbarPos("US", "Trackbar")
		u_v = cv.getTrackbarPos("UV", "Trackbar")


		l = np.array([l_h, l_s, l_v])
		u = np.array([u_h, u_s, u_v])
		mask_table_hsv = cv.inRange(frame_hsv, l, u)        

		res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)    
		show_imag1e("Frame", frame)
		show_imag1e("Mask", mask_table_hsv)
		show_imag1e("Res", res)
		if cv.waitKey(25) & 0xFF == ord('q'):
				break
	cv.destroyAllWindows()
img=cv.imread('1_01.jpg')

#_, thresh = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
#plt.imshow(img)
#plt.show()

find_color_values_using_trackbar(img)
exit()
low_yellow = (94, 7, 172)
high_yellow = (120, 146, 255)
#low_yellow = (0,127, 0)
#high_yellow = (255, 255, 255)
#low_yellow = (0,0, 0)
#high_yellow = (255, 238, 255)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
cv.imwrite('img_initial.jpg', img)
cv.imwrite('mask_yellow_hsv.jpg', mask_yellow_hsv)
cv.waitKey(0)
cv.destroyAllWindows()
exit()
img = cv2.imread('1_01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()
mser.setMinArea(100)
mser.setMaxArea(800)

coordinates, bboxes = mser.detectRegions(gray)

vis = img.copy()
coords = []
for coord in coordinates:
    bbox = cv2.boundingRect(coord)
    x,y,w,h = bbox
    if w< 10 or h < 10 or w/h > 5 or h/w > 5:
        continue
    coords.append(coord)

colors = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]

np.random.seed(0)
canvas1 = img.copy()
canvas2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
canvas3 = np.zeros_like(img)

for cnt in coords:
    xx = cnt[:,0]
    yy = cnt[:,1]
    color = colors[np.random.choice(len(colors))]
    canvas1[yy, xx] = color
    canvas2[yy, xx] = color
    canvas3[yy, xx] = color

cv2.imwrite("result1.png", canvas1)
cv2.imwrite("result2.png", canvas2)
cv2.imwrite("result3.png", canvas3)
