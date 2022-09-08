import cv2 as cv
import numpy as np
import white_blance as wb

def HSV_mask(img, hsv):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
 
    lower = np.array([hsv[0], hsv[2], hsv[4]])
    upper = np.array([hsv[1], hsv[3], hsv[5]])

    mask = cv.inRange(img_hsv, lower, upper)

    return mask

def find_direction(img, img2):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    total_cx = 0
    total_cy = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 100:
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        
            total_cx += cX
            total_cy += cY

            # cv.circle(img2, (cX, cY), 10, (1, 227, 254), -1)
            cv.drawContours(img2, cnt, -1, (0, 255, 0), 2)

    lane_cx = total_cx/2
    lane_cy = total_cy/2

    if lane_cx > int(img.shape[0]/2):
        print('turn right')
    elif lane_cx < int(img.shape[0]/2):
        print('turn left')
    else:
        print('go straghit')
    
    # cv.circle(img2, (int(lane_cx), int(lane_cy)), 10, (0, 0, 254), -1)
    return img2

hsv = [13, 28, 60, 160, 178, 255]

img = cv.imread('Ray_code0.jpg')
img_blur = cv.GaussianBlur(img,(5, 5), 0)

img_masked = HSV_mask(img_blur, hsv)
img_cnt = find_direction(img_masked, img_blur)

cv.imshow('result1', img_masked)
cv.imshow('result2', img_cnt)

cv.waitKey(0)

# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if ret:
#         # frame = wb.white_balance_3(frame)
#         img_blur = cv.GaussianBlur(frame, (5, 5), 0)
#         img_masked = HSV_mask(img_blur, hsv)
#         img_cnt = find_edge(img_masked, img_blur)

#         cv.imshow('video', img_cnt)

#     if cv.waitKey(27) & 0xFF == ord('q'):
#         break 

#     cv.waitKey(1)


# cap.release()
# cv.destroyAllWindows()
