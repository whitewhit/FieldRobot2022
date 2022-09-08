import cv2 as cv

cap =cv.VideoCapture(0)
i = 0
while(1):
    ret, frame = cap.read()
    k = cv.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('Ray_code' + str(i) + '.jpg', frame)
        i += 1
    cv.imshow('cap', frame)
cap.release()
cv.destroyAllWindows()