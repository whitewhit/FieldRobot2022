from inspect import Parameter
import cv2 as cv
import numpy as np

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5) , 0)
    edge = cv.Canny(blur, 150, 200)
    return edge

def region_of_interest(image): ##做一個梯形的遮罩
    hei = image.shape[0]
    wid = image.shape[1]

    triangle = np.array([(0, hei), 
                        # np.int32((wid / 2, (2 * hei) / 5)), 
                        (300, 480),
                        (600, 460),
                        (wid, hei)])
    
    mask = np.zeros_like(image)
    mask = cv.fillPoly(mask, [triangle], (255, 255, 255))
    
    mask_image = cv.bitwise_and(image, mask)
    return mask_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for  x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), 255, 2) 

    return line_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        Parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = Parameters[0]
        intercept = Parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(7/11))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


img = cv.imread('Lane.jpg')
img_copy = np.copy(img)
img_canny = canny(img)
img_mask = region_of_interest(img_canny)

lines = cv.HoughLinesP(img_mask, 1.0, np.pi/180, 100, np.array([]), minLineLength = 10, maxLineGap = 100) #image, rho, theta, threshod, lines
average_lines = average_slope_intercept(img_copy, lines)
line_img = display_lines(img, average_lines)
combo_img = cv.addWeighted(img_copy, 0.8, line_img, 1, 1)

cv.imshow('test', line_img)
cv.imshow('hi', img_mask)
cv.imshow('combo', combo_img)
cv.waitKey(0)
