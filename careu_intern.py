import cv2 as cv
import numpy as np

verboz = 1

def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.25, fy=0.25)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(image):
    image1 = image
    image_m_blur = cv.medianBlur(image, 1)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 1) 
    _, thresh = cv.threshold(image_g_blur, 30, 255, cv.THRESH_BINARY)

    if verboz == 1:
        show_image('image_thresholded', thresh)

    edges = cv.Canny(thresh, 200, 400)
    if verboz == 1:
        show_image('edges', edges)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    top_left, top_right, bottom_left, bottom_right = None, None, None, None
    for contour in contours:
        if len(contour) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contour.squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point
                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contour.squeeze(), axis=1)
            possible_top_right = contour.squeeze()[np.argmin(diff)]
            possible_bottom_left = contour.squeeze()[np.argmax(diff)]
            
            area = cv.contourArea(np.array([possible_top_left, possible_top_right, possible_bottom_right, possible_bottom_left]))
            if area > max_area:
                max_area = area
                top_left = possible_top_left
                top_right = possible_top_right
                bottom_left = possible_bottom_left
                bottom_right = possible_bottom_right

    if top_left is None:
        return "Corners not detected"

    width, height = 1700, 1700
    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    try:
        cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
        cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
        if verboz == 1:
            show_image("detected corners", image_copy)
    except Exception as e:
        return str(e)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)
    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
    show_image("Cropped table", result)
    
    return result

image = cv.imread('2_48.jpg')

low_red = np.array([0, 100, 100])
high_red = np.array([10, 255, 255])

img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
mask_red_hsv = cv.inRange(img_hsv, low_red, high_red)

extrage_careu(mask_red_hsv)

lines_horizontal = []
val = 1260
for i in range(0, val + 1, 90):
    l = [(0, i), (val - 1, i)]
    lines_horizontal.append(l)
print(lines_horizontal)

lines_vertical = []
for i in range(0, val + 1, 90):
    l = [(i, 0), (i, val - 1)]
    lines_vertical.append(l)

print(lines_vertical)
