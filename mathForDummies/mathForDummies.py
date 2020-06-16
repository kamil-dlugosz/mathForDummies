import sys
import cv2 as cv
import numpy as np

def loadImage(image_path):
    image = cv.imread(image_path, 0)
    if image is None:
        sys.exit('Source image not found, path: ' + image_path) 
    
    #image = np.copy(original_image)
    #height = 1000
    #width = round(height/image.shape[0]*image.shape[1])
    #return cv.resize(image, (width, height))
    #cv.imshow('image', image)
    #cv.waitKey(0)
    return image

def loadTemplates():
    templates = []
    for i in range(18):
        similar_templates = []
        for j in range(4):
            template_image = cv.imread('templates/' + str(j) + '/' + str(i) + '.png', 0)
            if template_image is None:
                sys.exit('Template image not found, path: ' + 'templates/' + str(j) + '/' + str(i) + '.png') 
            _, template = cv.threshold(template_image, 127, 255, cv.THRESH_BINARY)
            similar_templates.append(template)
        templates.append(similar_templates)   
    return templates

def loadEquations():
    equ = []
    equ.append(cv.imread('cropped/t2.png', 0))
    equ.append(cv.imread('cropped/t1.png', 0))
    equ.append(cv.imread('cropped/t3.png', 0))
    equ.append(cv.imread('cropped/t4.png', 0))
    equ.append(cv.imread('cropped/t5.png', 0))
    return equ
    
def findPageCorners(img):
    _, image = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    image = cv.medianBlur(image, 19)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations = 3, borderType = cv.BORDER_REPLICATE)    
    
    correct_lines = []
    canny_edges = cv.Canny(image, 50, 150, apertureSize = 3)
    hough_lines = cv.HoughLines(canny_edges, 1, np.pi/135, 250)
    for hline in hough_lines:
        isNew = True
        for cline in correct_lines:
            if (cline[0] + 200) > hline[0][0] > (cline[0] - 200) and (cline[1] + np.radians(5)) > hline[0][1] > (cline[1] - np.radians(5)):
                isNew = False
                break
        if isNew:
            correct_lines.append([hline[0][0], hline[0][1]])

    for line in correct_lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv.line(image, (x1, y1), (x2, y2), (200, 200, 200), 16)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 0), 10)    
    height = 1000
    width = round(height/image.shape[0]*image.shape[1])
    cv.imshow('image', cv.resize(image, (width, height)))
    cv.waitKey(0)
    
    if len(correct_lines) != 4:
        sys.exit('Invalid numer of page edges: ' + str(len(correct_lines))) 

    corners = []
    height, width = image.shape
    for i in range(3):
        for j in range(i, 4):
            theta1 = correct_lines[i][1]
            theta2 = correct_lines[j][1]
            if theta1 - np.radians(4) < theta2 < theta1 + np.radians(4):
                continue
            else:
                rho1 = correct_lines[i][0]
                rho2 = correct_lines[j][0]
                cos1 = np.cos(theta1)
                sin1 = np.sin(theta1)
                cos2 = np.cos(theta2)
                sin2 = np.sin(theta2)
                x = (rho2*sin1 - rho1*sin2)/(cos2*sin1 - cos1*sin2)
                y = rho1/sin1 - cos1/sin1*x
                if (x < 0 or x > width or y < 0 or y > height):
                    continue
                corners.append([x, y])

                #heightxd = 1000
                #widthxd = round(heightxd/image.shape[0]*image.shape[1])
                #cv.circle(image, (int(x), int(y)), 4, (150, 150, 150), 50)
                #cv.circle(image, (int(x), int(y)), 2, (0, 0, 0), 30)
                #cv.imshow('image', cv.resize(image, (widthxd, heightxd)))
                #cv.waitKey(0)
    if len(corners) != 4:
        sys.exit('Invalid numer of corners: ' + str(len(corners))) 
    return corners

def fixPerspective(source_image, corners):
    cnrs = np.array(corners)
    skewed_corners = np.zeros_like(cnrs)

    top = np.argmin(cnrs[:, 1], 0)
    bottom = np.argmax(cnrs[:, 1], 0)
    left = np.argmin(cnrs[:, 0], 0)
    right = np.argmax(cnrs[:, 0], 0)
    if ((cnrs[top][0] - cnrs[left][0])**2 + (cnrs[top][1] - cnrs[left][1])**2) > ((cnrs[top][0] - cnrs[right][0])**2 + (cnrs[top][1] - cnrs[right][1])**2):
        skewed_corners = np.array([cnrs[top], cnrs[right], cnrs[bottom], cnrs[left]])
    else:
        skewed_corners = np.array([cnrs[left], cnrs[top], cnrs[right], cnrs[bottom]])

    correct_corners = np.array([[0, 0], [3536, 0], [3536, 5000], [0, 5000]])
    homography, _ = cv.findHomography(skewed_corners, correct_corners)
    fixed_image = cv.warpPerspective(source_image, homography, (3537, 5001))
    
    heightxd = 1000
    widthxd = round(heightxd/fixed_image.shape[0]*fixed_image.shape[1])
    cv.imshow('image', cv.resize(fixed_image[200:4800, 150:3386], (widthxd, heightxd)))
    cv.waitKey(0)

    return fixed_image[200:4800, 150:3386].copy()

def cropEquations(source_image):
    binary_image = cv.adaptiveThreshold(source_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 10)
    binary_image = cv.medianBlur(binary_image, 7)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel, iterations = 2, borderType = cv.BORDER_REPLICATE)
    #binary_image = cv.dilate(binary_image, kernel)
    #heightxd = 1000
    #widthxd = round(heightxd/binary_image.shape[0]*binary_image.shape[1])
    #cv.imwrite('binary.png', binary_image)
    #cv.imshow('image', cv.resize(binary_image, (widthxd, heightxd)))
    #cv.waitKey(0)

    cropped_equations = []
    equ_top = 0
    equ_bottom = 0
    before_state = False
    for y in range(0, 4600, 10):
        print('loading: ' + str(y/46) + '%')
        is_found = False
        for x in range(0, 3236, 10):
            if binary_image[y, x] != 0:
                is_found = True
                break
        if is_found and before_state == False:
            equ_top = y - 25
        if is_found == False and before_state:
            equ_bottom = y + 25
            cropped_equations.append(binary_image[equ_top:equ_bottom].copy())
            #cv.imwrite('cropped/' + str(y) + '.png', binary_image[equ_top:equ_bottom])
        before_state = is_found
    return cropped_equations

def contourBoundaries(contour):
    left = 10000
    right = 0
    for point in contour:
        x = point[0][0]
        if left > x:
            left = x
        if right < x:
            right = x
    return (left, right)

def myMatchShapes(undef_char_cnt, templates, index):
    var_num = 1
    votes = np.zeros(var_num)
    for i in range(var_num):
        temp_cnt, _ = cv.findContours(templates[index][i], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        votes[i] = cv.matchShapes(undef_char_cnt, temp_cnt[0], 3, 0.0)
    return np.min(votes)

def recognizeChar(undef_char, templates):
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', '=', '(', '[']
    contours, _ = cv.findContours(undef_char, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv.contourArea(cnt) < 100:
            contours.remove(cnt)            
    if len(contours) < 1:
        return    
    
    undef_char_cnt = contours[0]
    _, _, _, height = cv.boundingRect(undef_char_cnt)
    cnt_area = cv.contourArea(undef_char_cnt)
    inf = 1000000.0
    char = 'm'
    if len(contours) > 1:
        if cnt_area < 1750:
            votes = np.array([[13, inf], [15, inf]])
            for i, temp_index in enumerate([13, 15]):
                votes[i][1] = myMatchShapes(undef_char_cnt, templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            print(votes)
            print('somewhere between: :  =           ', ' because len =', str(len(contours)), '  AND  ', 'area =', cnt_area)
            print('recognized char:', char)
        elif cnt_area > 1750:
            votes = np.array([(0, inf), (4, inf), (6, inf), (8, inf), (9, inf)])
            for i, temp_index in enumerate([0, 4, 6, 8, 9]):
                votes[i][1] = myMatchShapes(undef_char_cnt, templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            print(votes)
            print('somewhere between: 0  4  6  8  9  ', ' because len =', str(len(contours)), '  AND  ', 'area =', cnt_area)
            print('recognized char:', char)
    else:
        if height < 50:
            votes = np.array([[11, inf], [12, inf], [14, inf]])
            for i, temp_index in enumerate([11, 12, 14]):
                votes[i][1] = myMatchShapes(undef_char_cnt, templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            print(votes)
            print('somewhere between: -  *   ,       ', ' because len =', str(len(contours)), '  AND  ', 'area =', cnt_area)
            print('recognized char:', char)
        else:
            votes = np.array([[1, inf], [2, inf], [3, inf], [5, inf],
                              [7, inf], [10, inf], [16, inf], [17, inf]])
            for i, temp_index in enumerate([1, 2, 3, 5, 7, 10, 16, 17]):
                votes[i][1] = myMatchShapes(undef_char_cnt, templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            print(votes)
            print('somewhere between: 1 2 3 5 7 + ( [', ' because len =', str(len(contours)), '  AND  ', 'area =', cnt_area)
            print('recognized char:', char)

    #cv.imshow('img', cv.drawContours(np.zeros((200, 150), dtype = np.uint8), undef_char_cnt, -1, (200, 200, 200), 1))
    #cv.waitKey(0)
    #cv.imshow('img', undef_char)
    #cv.waitKey(0)
    return char

def recognizeEquation(cropped_equation, templates):
    contours, _ = cv.findContours(cropped_equation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda contours: contourBoundaries(contours)[0])

    index = 0

    equation_string = []
    for contour in sorted_contours:
        left, right = contourBoundaries(contour)
        undef_char = cropped_equation[:, left-2:right+2].copy()
        char = recognizeChar(undef_char, templates)
        if char == 'm' or char is None:
            continue
        equation_string.append(char)
        cv.rectangle(equation, (left, 0), (right, 200), (0, 0, 0), -1)

        chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', '=', '(', '[', '(', '[']
        print('correct char: ', chars[index])
        index = index + 1
        #cv.imshow('img', undef_char)
        #cv.waitKey(0)

    print(chars)
    return equation_string

Gimage = loadImage('equations/test.jpg')
#Gimage = loadImage('trash-bin/equations_old/2.png')
Gcorners = findPageCorners(Gimage)
Gfixed_image = fixPerspective(Gimage, Gcorners)
Gequations = cropEquations(Gfixed_image)
#Gequations = loadEquations()
Gtemplates = loadTemplates()
asdasd = []
for equation in Gequations:
    char_list = recognizeEquation(equation, Gtemplates)
    print(char_list)
    asdasd.append(char_list)

print(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', '=', '(', '[', '(', '['])
for asd in asdasd:
    print(asd)


    








    #bottom_c_i_1 = 0
    #bottom_c_i_2 = 0
    #right_c_i = 0
    #left_c_i = 0
    #for i in range(4):
    #    if corners[i][1] > corners[bottom_c_i_1][1]:
    #        bottom_c_i_2 = bottom_c_i_1
    #        bottom_c_i_2 = i
    #    if corners[i][0] > corners[right_c_i][1]:
    #        right_c_i = i
    #    if corners[i][0] < corners[left_c_i][1]:
    #        left_c_i = i
            
    #skewed_corners = np.zeros(4,2)
    #if right_c_i == bottom_c_i_1:
    #    skewed_corners[3] = corners[bottom_c_i_1]
    #    skewed_corners[2] = corners[bottom_c_i_2]
    #elif right_c_i == bottom_c_i_2:
    #    skewed_corners[3] = corners[bottom_c_i_2]
    #    skewed_corners[2] = corners[bottom_c_i_1]



    #skewed_corners[0] = corners[bottom_c_i_1]
    #skewed_corners[1] = corners[bottom_c_i_2]