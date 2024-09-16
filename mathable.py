import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
verboz = 1
def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.25,fy=0.25)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def extrage_careu(image):
    image1=image
    show_image("image1", image1)
    image_m_blur = cv.medianBlur(image,1)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 1) 
    _, thresh = cv.threshold(image_g_blur, 30, 255, cv.THRESH_BINARY)

    if verboz==1:
        show_image('image_thresholded',thresh) #

    edges =  cv.Canny(thresh ,200,400)	
    if verboz==1:
        show_image('edges',edges) #
    contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    max_area = 0
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left


    width = 1700
    height = 1700
    image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
    try:
        cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    except:
        return "error","error","error","error"
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    if verboz==1:
        show_image("detected corners",image_copy)
    #return top_left,top_right,bottom_left,bottom_right

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result,cv.COLOR_GRAY2BGR)
    show_image("Cropped table", result)
    
    return result, top_left,top_right,bottom_left,bottom_right

image=cv.imread('4_23.jpg')

low_yellow = (102,48,94)
high_yellow = (255, 255, 255)
img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
careu_exterior_cropped, top_left,top_right,bottom_left,bottom_right = extrage_careu(mask_yellow_hsv)
show_image("matrix cropped",careu_exterior_cropped)
plt.imshow(careu_exterior_cropped)
plt.show()
careu_interior_cropped = careu_exterior_cropped[234:1480, 230:1480]
show_image("inner matrix cropped",careu_interior_cropped)

low_yellow = (0,21,120)
high_yellow = (62, 120, 255)
img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
image_copy = cv.cvtColor(mask_yellow_hsv.copy(),cv.COLOR_GRAY2BGR)
width = 1700
height = 1700
puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")
M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
result = cv.warpPerspective(image_copy, M, (width, height))

poza_originala_cropped = result[220:1470, 220:1470]
show_image("ceva", poza_originala_cropped)

def classify_patch(patch, template_folder="/Users/dariadragomir/AI_siemens/Mathable/train/cifre/", threshold=0.8):
    best_match = None
    best_score = -1
    best_number = None

    patch_gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    print(patch_gray.shape)
    
    array ={}
    
    for template_file in os.listdir(template_folder):
        if template_file.endswith('.jpg'):
            template_number = int(template_file.split('.')[0])

            template_path = os.path.join(template_folder, template_file)
            template = cv.imread(template_path)
            template= cv.cvtColor(template,cv.COLOR_BGR2GRAY)
            print(template.shape)
            print(template_number)
            result = cv.matchTemplate(patch_gray, template, cv.TM_CCOEFF_NORMED)
            max_res = np.max(result)
    
            '''except:
                plt.imshow(patch_gray)
                plt.show()
                plt.imshow(template)
                plt.show()
                #print(patch_gray)
                print()
                #print(template)
            '''
            print(result)
            if template_number not in array:
                array[template_number] = max_res
    mx=-1
    cif=None
    for elem in array:
        if array[elem]>mx:
            mx=array[elem]
            cif=elem
    print(cif)



lines_horizontal=[]
val = 1260
for i in range(0,val+1,90):
    l=[]
    l.append((0,i))
    l.append((val-1,i))
    lines_horizontal.append(l)
print(lines_horizontal)
lines_vertical=[]
for i in range(0,val+1,90):
    l=[]
    l.append((i,0))
    l.append((i,val-1))
    lines_vertical.append(l)
    
print()
print(lines_vertical)

galben = [(0,0),(0,6),(0,7),(0,13),(6,0),(7,0),(6,13),(7,13),(13,0),(13,6),(13,7),(13,13)]
mov = [(1,1),(2,2),(3,3),(4,4),(9,9),(10,10),(11,11),(12,12),(12,1),(11,2),(10,3),(9,4),(1,12),(2,11),(3,10),(4,9)]
plus = [(3,6),(4,7),(6,4),(7,3),(9,6),(10,7),(6,10),(7,9)]
minus = [(2,5),(2,8),(5,2),(8,2),(11,5),(11,8),(5,11),(8,11)]
ori = [(3,7),(4,6),(6,3),(7,4),(9,7),(10,6),(6,9),(7,10)]
div = [(1,4),(1,9),(4,1),(9,1),(12,4),(12,9),(4,12),(9,12)]
unu = [(6,6)]
doi = [(6,7)]
trei = [(7,6)]
patru = [(7,7)]

m_initial = np.empty((14,14), dtype=object)
for i in range(len(m_initial)):
	for j in range(len(m_initial)):
		if (i,j) in galben:
			m_initial[i][j]='3x'
		elif (i,j) in mov:
			m_initial[i][j]='2x'
		elif (i,j) in plus:
			m_initial[i][j]='+'
		elif (i,j) in minus:
			m_initial[i][j]='-'
		elif (i,j) in ori:
			m_initial[i][j]='x'
		elif (i,j) in div:
			m_initial[i][j]='/'
		elif (i,j) in unu:
			m_initial[i][j]=1
		elif (i,j) in doi:
			m_initial[i][j]=2
		elif (i,j) in trei:
			m_initial[i][j]=3
		elif (i,j) in patru:
			m_initial[i][j]=4
		else:
			m_initial[i][j]='o'

copie_i = m_initial.copy()
#result = careul ramas doar din casete fara margine
#copie_i = creez harta fara nicio piesa pe ea ( x2-> matrice )
def determina_configuratie_careu_ocifre(img, thresh, lines_horizontal, lines_vertical, copie_i):
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            #+14 -14 +14 -14
            y_min = lines_vertical[j][0][0] 
            y_max = lines_vertical[j + 1][1][0] 
            x_min = lines_horizontal[i][0][1]  
            x_max = lines_horizontal[i + 1][1][1]
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_orig=img[x_min:x_max, y_min:y_max].copy()
            patch_orig= cv.cvtColor(patch_orig,cv.COLOR_BGR2GRAY)
            Medie_patch=np.mean(patch) 
            if Medie_patch>100: #este o piesa pusa
                print("cifra")
                plt.imshow(patch)
                plt.show()
                classified_number = classify_patch(patch, template_folder="/Users/dariadragomir/AI_siemens/Mathable/train/cifre_cropped/", threshold=0.8)
                print(classified_number)
determina_configuratie_careu_ocifre(poza_originala_cropped, poza_originala_cropped, lines_horizontal, lines_vertical, copie_i)
