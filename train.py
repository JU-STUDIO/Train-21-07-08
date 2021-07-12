import numpy as np
import cv2
from matplotlib import pyplot as plt

def videoCapture():
    file_path = r'./Video/AVSS_E2.avi'
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('FRAME  ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('WIDTH  ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('HEIGHT ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    cv2.destroyAllWindows()

def mask():
    pathFile= r'./Frames/FrameNo0.png'

    imgOriginal = cv2.imread(pathFile)
    cv2.imshow('Original',imgOriginal)

    width = 720
    height = 576

    col_interval=50
    for i in range(0, width, col_interval):
        cv2.line(imgOriginal,(i,0),(i,height),(255,255,0),2)
        cv2.putText(imgOriginal,'%s'%(i),(i,int(col_interval/2)),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),thickness=1)
    cv2.imshow('Make_Col',imgOriginal)

    row_interval=50
    for i in range(0, height, row_interval):
        cv2.line(imgOriginal,(0,i),(width,i),(255,255,0),2)
        cv2.putText(imgOriginal,'%s'%(i),(int(row_interval/2),i),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),thickness=1)
    
    cv2.imshow('Nake_Row',imgOriginal)

    mask = np.zeros(imgOriginal.shape[:2],dtype='uint8')
    cv2.imshow('Mask',mask)

    pts = np.array([[280,80],[0,300],[0,500],[700,500],[700,80]],np.int32)
    
    cv2.polylines(imgOriginal,[pts],True,(255,0,0),thickness=2)
    cv2.imshow('AreaOfInterest',imgOriginal)

    cv2.fillPoly(mask,[pts],255,1)
    cv2.imshow('Masked',mask)

    masked = cv2.bitwise_and(imgOriginal, imgOriginal, mask=mask)
    cv2.imshow('MaskedImg',masked)

    plt.figure(figsize=(20.1,20.1))
    plt.subplot(231),plt.imshow(imgOriginal),plt.title('Original_2'),plt.xticks([]),plt.yticks([])
    plt.subplot(232),plt.imshow(masked),plt.title('MaskedImg_2'),plt.xticks([]),plt.yticks([])
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

mask()
