import numpy as np
import cv2

file_path = r'C:\\Users\\NTUS\\Desktop\\20210708\\AbandonObjVideo\\AVSS_E2.avi'
cap = cv2.VideoCapture(file_path)

counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == 0:
        break

    frameNo = 'D:/Frames/' + 'FrameNo' + str(counter) + '.png'
    cv2.imwrite(frameNo,frame)
    counter = counter + 1

    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

print('FRAME  ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('WIDTH  ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('HEIGHT ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()
cv2.destoryAllWindows()