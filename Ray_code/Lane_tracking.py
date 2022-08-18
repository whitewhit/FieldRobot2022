import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
        for line in lines:
            x1, x2, y1, y2, = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), 255, 2) 

    return line_image

img = cv.imread('Lane.jpg')
img_canny = canny(img)
img_mask = region_of_interest(img_canny)

lines = cv.HoughLinesP(img_mask, 1.0, np.pi/180, 100, np.array([]), minLineLength = 10, maxLineGap = 1) #image, rho, theta, threshod, lines
line_img = display_lines(img, lines)

cv.imshow('test', line_img)
cv.imshow('hi', img_mask)
cv.waitKey(0)
# plt.imshow(img_canny)
# plt.show()
