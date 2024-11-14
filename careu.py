import cv2 as cv
import numpy as np
verboz = 1
def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.25,fy=0.25)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def extrage_careu(image):
    image1=image
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
    
    return result


image=cv.imread('1_01.jpg')

low_yellow = (0,0,0)
high_yellow = (122, 255, 116)
img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
extrage_careu(mask_yellow_hsv)

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
'''
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
        #if Medie_patch>valoare_anume: este o piesa pusa
            #clasific cifra cu un trashold 
        
result = careul ramas doar din casete fara margine
copie_i = creez harta fara nicio piesa pe ea ( x2-> matrice )
update copie_i pt fiecare mutare 
determina_configuratie_careu_ocifre(result,result,lines_horizontal,lines_vertical, copie_i)

-- creez copie_i = careul fara nicio piesa pe el pt reguli 
-- gasesc valori de LY HY pt coordonate careu exterior
-- gasit coordonate careu interior 
-- aplic wrap si perspective din extrage_careu() ca sa centreze careul interior
-- gasit LY HY pt contrast bun intre piesele puse(albe) si backgroung(negru)
-- cod pt determina_configuratie_careu_ocifre()
-- fac dataset pt cifre (folder de cifre)
-- tot in determina_configuratie_careu_ocifre() scriu functie clasificare cifra cu MatchTemplate din cv 
-- calculez scor
'''

'''
def get_templates1(img,thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='str')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0] 
            x_min = lines_horizontal[i][0][1] 
            x_max = lines_horizontal[i + 1][1][1] 
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_orig=img[x_min:x_max, y_min:y_max].copy()
            #plt.imshow(patch_orig)
            #plt.show()
            cv.imwrite("templates/"+str(i)+"_"+str(j)+".jpg",patch_orig)
            print(i)
    return matrix

'''
