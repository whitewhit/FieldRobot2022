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
    count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 50:
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cY)
            total_cx += cX
            total_cy += cY
            count += 1
            cv.circle(img2, (cX, cY), 10, (1, 227, 254), -1)
            cv.drawContours(img2, cnt, -1, (0, 255, 0), 2)

    lane_cx = total_cx/count
    lane_cy = total_cy/count
    
    if lane_cx < int(img.shape[1]/2):
        print('turn left     ', lane_cx, '   ', img.shape[1]/2)
    elif lane_cx > int(img.shape[1]/2):
        print('turn right    ', lane_cx, '   ', img.shape[1]/2)

    else:
        print('go straghit')
    
    cv.circle(img2, (int(lane_cx), int(lane_cy)), 10, (0, 0, 255), -1)
    return img2

def region_of_interest(image,x,y): 
    hei = image.shape[0]
    wid = image.shape[1]

    triangle = np.array([np.int32((0, hei*y)), 
                        # np.int32((wid / 2, (2 * hei) / 5)), 
                        np.int32((wid, hei*y)),
                        np.int32((wid, hei)),
                        np.int32((0, hei))])
    
    mask = np.zeros_like(image)
    mask = cv.fillPoly(mask, [triangle], (255, 255, 255))
    
    mask_image = cv.bitwise_and(image, mask)
    return mask_image

# hsv = [41,65, 115, 255, 10, 70]
# hsv = [40, 53, 60, 255, 30, 70] #for camera
hsv = [0, 179, 0, 73, 0, 110] #for phone camera

# img = cv.imread('Ray_code0.jpg')
# img_blur = cv.GaussianBlur(img,(5, 5), 0)

# img_masked = HSV_mask(img_blur, hsv)
# img_roi = region_of_interest(img_masked,1,2/3)
# img_dilate = cv.dilate(img_roi,(5,5),iterations = 1) 
# img_cnt = find_direction(img_dilate, img_blur)

# cv.imshow('result1', img_masked)
# cv.imshow('hi', img_dilate)
# cv.imshow('result2', img_cnt)

# cv.waitKey(0)

cap = cv.VideoCapture('20220908_205918.mp4')
# cap = cv.VideoCapture('20220908_205946.mp4')
while True:
    ret, frame = cap.read()
    if ret:
        # frame = wb.white_balance_3(frame)
        frame = cv.resize(frame,(0, 0), fx = 0.25, fy = 0.25)
        img_blur = cv.GaussianBlur(frame, (5, 5), 0)
        img_mask = HSV_mask(img_blur, hsv)
        img_roi = region_of_interest(img_mask,1,2/3)
        img_dilate = cv.dilate(img_roi,(5,5),iterations = 1) 
        img_cnt = find_direction(img_dilate, img_blur)

        print(frame.shape)
        cv.imshow('result1', img_mask)
        cv.imshow('hi', img_dilate)
        cv.imshow('result2', img_cnt) 
    
    else:
        break

    if cv.waitKey(27) & 0xFF == ord('q'):
        break 

    cv.waitKey(100)


cap.release()
cv.destroyAllWindows()
