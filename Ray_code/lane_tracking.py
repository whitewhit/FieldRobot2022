import cv2 as cv
import numpy as np
import PID as pid
import serial    #導入serial庫
import time


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
            total_cx += cX
            total_cy += cY
            count += 1
            cv.circle(img2, (cX, cY), 10, (1, 227, 254), -1)
            cv.drawContours(img2, cnt, -1, (0, 255, 0), 2)
    if count ==0:
        return img2
    else :
        lane_cx = total_cx/count
        lane_cy = total_cy/count
    

    ser.flush()
    if lane_cx < int(img.shape[1]/2):
        a = A.update(int(img.shape[1]/2) - lane_cx)
        if a != None:
            print('feedback = ',  int(a), 'Turn right')
            ser.write(b'%d\n' %int(a))
        
    elif lane_cx > int(img.shape[1]/2):
        b = A.update(int(img.shape[1]/2) - lane_cx)
        if b != None:
            print('feedback = ',  int(b), 'Turn left')
            ser.write(b'%d\n' %int(b))
    
    
        

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

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)   #打開端口，每一秒返回一個消息
#try模塊用來結束循環（靠拋出異常）
    A = pid.PID(0.1, 0.1, 0.4)
    # hsv = [40, 53, 60, 255, 30, 70] #for camera
    hsv = [33, 130, 106, 182, 0, 239] #for phone camera

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
#         frame = cv.resize(frame,(0, 0), fx = 0.25, fy = 0.25)
        img_blur = cv.GaussianBlur(frame, (5, 5), 0)
        img_mask = HSV_mask(img_blur, hsv)
        img_roi = region_of_interest(img_mask,1,2/3)
        img_dilate = cv.dilate(img_roi,(5,5),iterations = 1)
        img_cnt = find_direction(img_dilate, img_blur)
 
        cv.imshow('result1', img_mask)
        cv.imshow('hi', img_dilate)
        cv.imshow('result2', img_cnt)
        
        try:
#             ser.write(b'r\n')
            response = ser.readline().decode().rstrip()#用response讀取端口的返回值
            print(response)
        except:
            print('error')
            
        
        if cv.waitKey(1) & 0xff == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()
#     try:
#         while True:
#             ser.write(b'r\n')
#             response = ser.readline().decode().rstrip()#用response讀取端口的返回值
#             print(response)#進行打印
#     
#     
#     except:
#         print('error')
#         ser.close()#拋出異常後關閉端口
    