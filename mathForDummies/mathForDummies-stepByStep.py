import sys
import cv2 as cv
import numpy as np

def loadImage(image_path):
    image = cv.imread(image_path, 0)
    if image is None:
        sys.exit('Source image not found, path: ' + image_path) 
    
    height = 1000
    width = round(height/image.shape[0]*image.shape[1])
    resized_image = cv.resize(image, (width, height))
    cv.imshow('image', resized_image)
    cv.waitKey(0)
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
    
def findPageCorners(source_image):
    _, image = cv.threshold(source_image, 127, 255, cv.THRESH_BINARY)
    image = cv.medianBlur(image, 5)

    image2 = image.copy()
    height = 1000
    width = round(height/image2.shape[0]*image2.shape[1])
    cv.imshow('image', cv.resize(image2, (width, height)))
    cv.waitKey(0)   
    
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key = lambda x: cv.contourArea(x))
    epsilon = 0.01
    while True:
        corners = np.array(cv.approxPolyDP(max_cnt, epsilon * cv.arcLength(max_cnt, True), True))
        corners = corners[:, 0]
        #print(corners)
        if len(corners) > 4:
            epsilon = epsilon + 0.02 
        elif len(corners) < 4:
            epsilon = epsilon - 0.02
        else:
            break
    
    #print(corners[:, 0])
    image2 = np.zeros_like(image, dtype = np.uint8)
    image2 = cv.drawContours(image2, [corners], -1, (255, 255, 255), 20)
    height = 1000
    width = round(height/image2.shape[0]*image2.shape[1])
    cv.imshow('image', cv.resize(image2, (width, height)))
    cv.waitKey(0)   
     
    return corners

def fixPerspective(source_image, cnrs):
    skewed_corners = np.zeros_like(cnrs)
    top = np.argmin(cnrs[:, 1], 0)

    if ((cnrs[top][0] - cnrs[(top+1)%4][0])**2 + (cnrs[top][1] - cnrs[(top+1)%4][1])**2) > ((cnrs[top][0] - cnrs[(top+3)%4][0])**2 + (cnrs[top][1] - cnrs[(top+3)%4][1])**2):
        skewed_corners = np.array([cnrs[top], cnrs[(top+3)%4], cnrs[(top+2)%4], cnrs[(top+1)%4]])
    else:
        skewed_corners = np.array([cnrs[(top+1)%4], cnrs[top], cnrs[(top+3)%4], cnrs[(top+2)%4]])
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
    binary_image = cv.dilate(binary_image, kernel)
    heightxd = 1000
    widthxd = round(heightxd/binary_image.shape[0]*binary_image.shape[1])
    cv.imshow('image', cv.resize(binary_image, (widthxd, heightxd)))
    cv.waitKey(0)

    cropped_equations = []
    equ_top = 0
    equ_bottom = 0
    before_state = False
    for y in range(0, 4600, 10):
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
            cv.imwrite(str(y) + '.png', binary_image[equ_top:equ_bottom])
        before_state = is_found
    return cropped_equations

def myMatchShapes(undef_char_cnt, templates, index):
    var_num = 1
    votes = np.zeros(var_num)
    for i in range(var_num):
        temp_cnt, _ = cv.findContours(templates[index][i], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        votes[i] = cv.matchShapes(undef_char_cnt, temp_cnt[0], 3, 0.0)
    return np.min(votes)

def contourCenter(contour):
    n = len(contour)
    x = 0
    y = 0
    for point in contour:
        x = x + point[0][0]
        y = y + point[0][1]
    return (x//n, y//n)

def recognizeChar(undef_char, templates):
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', '=']
    undef_char_cnt, _ = cv.findContours(undef_char, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in undef_char_cnt:
        if cv.contourArea(cnt) < 100:
            undef_char_cnt.remove(cnt)            
    if len(undef_char_cnt) < 1:
        return    
    cv.imshow('img', cv.drawContours(np.zeros((200, 150), dtype = np.uint8), undef_char_cnt, -1, (200, 200, 200), 1))
    cv.waitKey(0)
    cv.imshow('img', undef_char)
    cv.waitKey(0)
    
    x, y, _, height = cv.boundingRect(undef_char_cnt[0])
    cnt_area = cv.contourArea(undef_char_cnt[0])
    inf = 1000000.0
    char = 'e'
    if len(undef_char_cnt) > 1:
        if cnt_area < 1750:
            votes = np.array([[13, inf], [15, inf]])
            for i, temp_index in enumerate([13, 15]):
                votes[i][1] = myMatchShapes(undef_char_cnt[0], templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            return char
        elif len(undef_char_cnt) == 3:
            char = chars[8]
            return char
        elif cnt_area > 1750:
            _, h = contourCenter(undef_char_cnt[1])
            if h-y < 0.4*height:
                char = chars[9]
                return char
            elif h-y > 0.6*height:
                char = chars[6]
                return char
            elif cv.contourArea(undef_char_cnt[1]) > 1200:
                char = chars[0]
                return char
            else:
                char = chars[4]
                return char
    else:
        if height < 50:
            votes = np.array([[11, inf], [12, inf], [14, inf]])
            for i, temp_index in enumerate([11, 12, 14]):
                votes[i][1] = myMatchShapes(undef_char_cnt[0], templates, temp_index)
            char = chars[int(votes[np.argmin(votes[:, 1]), 0])]
            return char
        else:
            epsilon = 0.018
            while True:
                poly = cv.approxPolyDP(undef_char_cnt[0], epsilon*cv.arcLength(undef_char_cnt[0], True), True)
                hull_ind = cv.convexHull(poly, returnPoints = False)
                defects = cv.convexityDefects(poly, hull_ind)
                if len(defects) == 4:
                    char = chars[10]
                    return char
                elif len(defects) == 1:
                    char = chars[7]
                    return char
                elif len(defects) > 2:
                    epsilon = epsilon + 0.001
                else:
                    break

            left_def = defects[np.argmin(poly[defects[:, 0, 1], 0, 0]), 0]
            right_def = defects[np.argmax(poly[defects[:, 0, 1], 0, 0]), 0]

            if poly[left_def[2], 0, 0] > poly[right_def[2], 0, 0] + 5:
                if poly[left_def[2], 0, 1] < poly[right_def[2], 0, 1]:
                    char = chars[2]
                    #return char
                else:
                    char = chars[5]
                    #return char
                        
            hull_image = np.zeros_like(undef_char, dtype = np.uint8)
            cv.drawContours(hull_image, undef_char_cnt, -1, (100, 100, 100), 1)
            hull_points = cv.convexHull(poly)
            hull_image = cv.applyColorMap(hull_image, cv.COLORMAP_BONE)
            cv.drawContours(hull_image, [poly], -1, (0, 255, 0), 1)
            cv.drawContours(hull_image, [hull_points], -1, (0, 0, 255), 1)
            for defect in defects:
                if defect[0][3] > 0:
                    cv.circle(hull_image, tuple(poly[defect[0][0]][0]), 3, (15, 15, 255), 1)
                    cv.circle(hull_image, tuple(poly[defect[0][1]][0]), 3, (255, 15, 15), 1)
                    cv.circle(hull_image, tuple(poly[defect[0][2]][0]), 3, (15, 255, 15), 1)
            cv.imshow("img", hull_image)
            cv.waitKey(0)
                
            canny = np.zeros_like(undef_char, dtype = np.uint8)
            canny = cv.Canny(undef_char, 50, 150, apertureSize = 3)
            lines = cv.HoughLines(canny, 3, 2 * np.pi / 180, 60)
            if lines is not None:
                char = chars[1]
                return char
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 10000 * (-b))
                    y1 = int(y0 + 10000 * (a))
                    x2 = int(x0 - 10000 * (-b))
                    y2 = int(y0 - 10000 * (a))
                    cv.line(canny, (x1, y1), (x2, y2), (200, 200, 200), 1)
            else:
                char = chars[3]
                return char
            cv.imshow("img", canny)
            cv.waitKey(0)
    return char

def recognizeEquation(cropped_equation, templates):
    contours, _ = cv.findContours(cropped_equation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda contour: cv.boundingRect(contour)[0])

    equation_string = []
    for contour in sorted_contours:
        left, _, width, _ = cv.boundingRect(contour)
        undef_char = cropped_equation[:, left-2:left+width+2].copy()
        char = recognizeChar(undef_char, templates)
        if char == 'm' or char is None:
            continue
        equation_string.append(char)
        image = equation.copy()
        image = cv.line(image, (left, 0), (left, 200), (100, 200, 100), 5)
        image = cv.line(image, (left+width, 0), (left+width, 200), (100, 200, 100), 5)
        image = cv.applyColorMap(image, cv.COLORMAP_BONE)
        widthxd = 1000
        heightxd = round(widthxd/image.shape[1]*image.shape[0])
        image = cv.resize(image, (widthxd, heightxd))
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.rectangle(equation, (left, 0), (left+width, 200), (0, 0, 0), -1)

    return equation_string

Gimage = loadImage('equations/a1.jpg')
Gcorners = findPageCorners(Gimage)
Gfixed_image = fixPerspective(Gimage, Gcorners)
Gequations = cropEquations(Gfixed_image)
#Gequations = loadEquations()
Gtemplates = loadTemplates()
for equation in Gequations:
    char_list = recognizeEquation(equation, Gtemplates)
    print(char_list)
    string = ''
    for char in char_list:
        if char == '=':
            continue
        string = string + char
    print(string, "\nDo you accept equation? y - yes, other - no")
    agreement = input()
    if agreement == 'y':
        print(eval(string))
